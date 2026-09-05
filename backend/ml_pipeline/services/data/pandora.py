"""
Dependecies: pip install pandas pyarrow
\backend\ml_pipeline\services\data\pandora.py
PANDORA Data Ingestion Pipeline
================================

Loads the PANDORA Big Five parquet export (jingjietan/pandora-big5 schema:
O, C, E, A, N, ptype, text, __index_level_0__), groups comments by user,
and prepares cleaned comment data for downstream Q-learning selection.

Why "grouping" isn't a plain author column here
-------------------------------------------------
The original PANDORA corpus (Gjurkovic & Snajder, 2021) joins comments.csv
to author_profiles.csv on Reddit username. This parquet export has already
broadcast each author's Big Five scores onto every one of their comments
and dropped the username, leaving only a row index and a `ptype` bucket
(a 32-class personality-stratification label, not an identity).

Since O/C/E/A/N are constant per author (same 5 scores repeated across
every comment that author wrote), the tuple (O, C, E, A, N) is used here
as a proxy user key: rows sharing all five scores are treated as one user.
This mirrors the "constant label per author" structure the dataset itself
relies on. It is a heuristic, not a cryptographic guarantee -- two authors
could in theory share all five rounded percentile scores. Any row whose
trait tuple does not repeat anywhere else in the file naturally ends up as
a group of one, which is the built-in fallback: no separate "ungrouped"
code path is needed, a singleton group already behaves like no grouping.

If literal Reddit usernames are required, request the original gated
distribution at https://psy.takelab.fer.hr/datasets/all/pandora and join
comments.csv / author_profiles.csv on `author` instead of using this
module's proxy key.

Scope
-----
This module owns ingestion ONLY:
    load PANDORA parquet -> group by (proxy) user -> clean -> drop
    unusable -> persist

Cleaning logic is NOT duplicated here. All text normalization goes through
the existing backend/ml_pipeline/cleaning/cleaner.py::DataCleaner, the same
cleaner used by the X ingestion path. This module just adapts PANDORA's
flat per-comment rows into the RawXData shape DataCleaner expects, and
maps the cleaned results back to their owning (proxy) user.

Q-learning does not load files or fetch PANDORA itself; it only consumes
the list[PreparedUserComments] returned by load_pandora_comments().

    comments = load_pandora_comments("<PANDORA_FILE>")
    selected_comments = q_learning.select(comments)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from backend.ml_pipeline.cleaning.cleaner import CleanedContent, DataCleaner, RawXData
from backend.ml_pipeline.services.data.quality import exclusion, safe_rate

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("backend/ml_pipeline/data/prepared/pandora_prepared.json")
_LAST_INGESTION_QUALITY: dict[str, Any] | None = None

# The five Big Five trait columns, in the order the dataset uses them.
_TRAIT_COLS = ("O", "C", "E", "A", "N")

GroupBy = Literal["traits", "row"]


@dataclass
class UserTraits:
    """Big Five percentile scores + stratification bucket for a (proxy) user."""

    O: float  # noqa: E741 - matches dataset column name
    C: float
    E: float
    A: float
    N: float
    ptype: int


@dataclass
class PreparedUserComments:
    """Cleaned, grouped comment data for a single PANDORA (proxy) user."""

    user_id: str
    traits: UserTraits | None
    comments: list[CleanedContent] = field(default_factory=list)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "traits": asdict(self.traits) if self.traits else None,
            "comments": [asdict(c) for c in self.comments],
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_raw_rows(pandora_file: str | Path) -> Iterable[dict[str, Any]]:
    """
    Stream rows out of the PANDORA parquet export.

    Expects columns: O, C, E, A, N, ptype, text (the jingjietan/pandora-big5
    schema). Requires pandas + pyarrow (`pip install pandas pyarrow`).
    """
    path = Path(pandora_file)
    if not path.exists():
        raise FileNotFoundError(f"PANDORA file not found: {path}")

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading the PANDORA parquet export requires pandas and pyarrow: "
            "pip install pandas pyarrow"
        ) from exc

    df = pd.read_parquet(path, columns=[*_TRAIT_COLS, "ptype", "text"])
    for row in df.to_dict(orient="records"):
        yield row


def _row_key(row: dict[str, Any]) -> str:
    """Proxy user id: the Big Five trait tuple, formatted as a stable string."""
    return "_".join(str(row.get(col)) for col in _TRAIT_COLS)


def _group_rows(
    rows: Iterable[dict[str, Any]],
    group_by: GroupBy,
) -> tuple[dict[str, tuple[UserTraits | None, list[str]]], dict[str, Any]]:
    """
    Group raw comment text by proxy user id.

    group_by="traits" (default): group by the (O, C, E, A, N) tuple, since
        that tuple is constant per author in this dataset. Rows with a
        trait tuple that appears only once naturally form a group of one --
        that IS the fallback, no extra branch is needed.
    group_by="row": skip grouping entirely; every row is its own unit
        (content_id becomes the row index). Use this if trait-tuple
        collisions between distinct authors are a concern for your use case.
    """
    grouped: dict[str, tuple[UserTraits | None, list[str]]] = {}
    skipped = 0
    invalid_traits = 0
    raw_rows = 0

    for i, row in enumerate(rows):
        raw_rows += 1
        text = row.get("text")
        if not text or not str(text).strip():
            skipped += 1
            continue

        if group_by == "row":
            key = f"row{i}"
        else:
            key = _row_key(row)

        traits = None
        try:
            traits = UserTraits(
                O=row["O"], C=row["C"], E=row["E"], A=row["A"], N=row["N"],
                ptype=int(row["ptype"]),
            )
        except (KeyError, TypeError, ValueError):
            invalid_traits += 1
            logger.debug("Row %d missing/invalid trait columns; storing without traits.", i)

        if key not in grouped:
            grouped[key] = (traits, [])
        grouped[key][1].append(str(text))

    if skipped:
        logger.warning("Skipped %d PANDORA rows with empty/missing text.", skipped)

    n_singletons = sum(1 for _, texts in grouped.values() if len(texts) == 1)
    comments_grouped = sum(len(t) for _, t in grouped.values())
    logger.info(
        "Grouped %d rows into %d proxy users (%d are singleton groups).",
        skipped + comments_grouped,
        len(grouped),
        n_singletons,
    )
    row_stats = {
        "raw_rows": raw_rows,
        "empty_or_missing_text": skipped,
        "missing_or_invalid_traits": invalid_traits,
        "rows_retained_for_grouping": raw_rows - skipped,
        "users_after_grouping": len(grouped),
        "comments_after_grouping": comments_grouped,
        "singleton_users": n_singletons,
        "group_by": group_by,
    }
    return grouped, row_stats


# ---------------------------------------------------------------------------
# Cleaning (delegated to existing DataCleaner) + regrouping by user
# ---------------------------------------------------------------------------

def _clean_all(
    grouped: dict[str, tuple[UserTraits | None, list[str]]],
    cleaner: DataCleaner,
) -> tuple[list[PreparedUserComments], dict[str, Any]]:
    """
    Flatten every user's comments into a single ordered payload, run it
    through the existing DataCleaner exactly once (so dedup / length
    filtering behave the same as they do for the X pipeline), then use a
    synthetic per-comment id to route cleaned items back to their user.

    No cleaning/normalization logic lives in this function or module --
    DataCleaner.clean() is the sole source of truth for that.
    """
    flat_posts: list[dict[str, Any]] = []
    idx_to_user: dict[int, str] = {}

    for user_id, (_traits, texts) in grouped.items():
        for text in texts:
            i = len(flat_posts)
            idx_to_user[i] = user_id
            # id_str must be a truthy, unique string so DataCleaner uses it
            # (not a text hash) as the basis for content_id, letting us
            # deterministically map cleaned items back to their user below.
            flat_posts.append({"text": text, "id_str": f"p{i}"})

    raw = RawXData(profile={}, tweets=flat_posts, replies=[])
    cleaned_items = cleaner.clean(raw)

    grouped_prepared: dict[str, list[CleanedContent]] = defaultdict(list)
    accepted_tweets = 0
    unmapped = 0
    for item in cleaned_items:
        if item.content_type != "tweet":
            continue  # skip the (absent) profile item; we only fed tweets
        accepted_tweets += 1
        suffix = item.content_id.rsplit("_p", 1)
        if len(suffix) != 2 or not suffix[1].isdigit():
            unmapped += 1
            logger.debug("Could not map cleaned item %s back to a user; dropping.", item.content_id)
            continue
        idx = int(suffix[1])
        user_id = idx_to_user.get(idx)
        if user_id is not None:
            grouped_prepared[user_id].append(item)
        else:
            unmapped += 1

    prepared = [
        PreparedUserComments(
            user_id=user_id,
            traits=grouped[user_id][0],
            comments=items,
        )
        for user_id, items in grouped_prepared.items()
        if items  # drop users left with zero usable comments after cleaning
    ]
    comments_retained = sum(len(u.comments) for u in prepared)
    account = cleaner.last_account or {}
    clean_stats = {
        "comments_before_cleaning": len(flat_posts),
        "users_before_cleaning": len(grouped),
        "cleaner": account,
        "accepted_tweets": accepted_tweets,
        "excluded_unmapped": unmapped,
        "comments_retained": comments_retained,
        "users_after_cleaning": len(prepared),
        "users_excluded_no_usable_comments": len(grouped) - len(prepared),
        "min_text_length": cleaner.min_text_length,
    }
    return prepared, clean_stats


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_prepared(prepared: list[PreparedUserComments], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [u.to_serializable() for u in prepared]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Saved prepared PANDORA data to %s", path)


def quality_sidecar_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}_quality.json")


def get_last_ingestion_quality() -> dict[str, Any] | None:
    """Quality measured by the most recent load_pandora_comments() call, if any."""
    return _LAST_INGESTION_QUALITY


def load_ingestion_quality(output_path: str | Path) -> dict[str, Any] | None:
    path = quality_sidecar_path(output_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _build_ingestion_quality(
    row_stats: dict[str, Any],
    clean_stats: dict[str, Any],
    *,
    source: str,
    min_text_length: int,
) -> dict[str, Any]:
    raw_rows = int(row_stats["raw_rows"])
    empty = int(row_stats["empty_or_missing_text"])
    retained_rows = int(row_stats["rows_retained_for_grouping"])
    before = int(clean_stats["comments_before_cleaning"])
    account = clean_stats.get("cleaner") or {}
    too_short = int(account.get("excluded_too_short", 0))
    duplicate = int(account.get("excluded_duplicate", 0))
    invalid = int(account.get("excluded_empty_or_invalid_post", 0))
    unmapped = int(clean_stats["excluded_unmapped"])
    retained = int(clean_stats["comments_retained"])
    users_before = int(clean_stats["users_before_cleaning"])
    users_after = int(clean_stats["users_after_cleaning"])

    exclusion_reasons = {
        "empty_or_missing_text": exclusion(
            empty,
            "Row text was missing or whitespace-only; dropped before grouping.",
            stage="ingestion",
        ),
        "missing_or_invalid_traits": exclusion(
            int(row_stats["missing_or_invalid_traits"]),
            "O/C/E/A/N or ptype missing/invalid. Row was kept for grouping; "
            "user is stored without traits and later excluded from the experiment sample.",
            stage="ingestion",
        ),
        "too_short": exclusion(
            too_short,
            f"cleaned_text length < {min_text_length} after DataCleaner normalization.",
            stage="cleaning",
        ),
        "duplicate_cleaned_text": exclusion(
            duplicate,
            "Identical cleaned_text already accepted in this DataCleaner.clean() pass.",
            stage="cleaning",
        ),
        "invalid_content": exclusion(
            invalid,
            "Post payload was empty; DataCleaner._clean_post returned None.",
            stage="cleaning",
        ),
        "unmapped_after_clean": exclusion(
            unmapped,
            "Cleaned tweet could not be mapped back to its proxy user.",
            stage="cleaning",
        ),
        "users_no_usable_comments": exclusion(
            int(clean_stats["users_excluded_no_usable_comments"]),
            "Proxy user had zero comments left after cleaning.",
            stage="cleaning",
        ),
    }

    return {
        "available": True,
        "source": source,
        "raw_rows": raw_rows,
        "rows_removed": empty,
        "rows_retained_for_grouping": retained_rows,
        "row_retention": safe_rate(
            retained_rows, raw_rows, denominator_name="raw_rows",
        ),
        "empty_or_missing_text": exclusion_reasons["empty_or_missing_text"],
        "missing_or_invalid_traits": exclusion_reasons["missing_or_invalid_traits"],
        "users_before_cleaning": users_before,
        "users_after_cleaning": users_after,
        "comments_before_cleaning": before,
        "comments_after_cleaning": retained,
        "comments_retained": retained,
        "comment_retention": safe_rate(
            retained, before, denominator_name="comments_before_cleaning",
        ),
        "excluded_too_short": exclusion_reasons["too_short"],
        "excluded_duplicate": exclusion_reasons["duplicate_cleaned_text"],
        "excluded_invalid_content": exclusion_reasons["invalid_content"],
        "excluded_unmapped": exclusion_reasons["unmapped_after_clean"],
        "users_excluded_no_usable_comments": exclusion_reasons["users_no_usable_comments"],
        "exclusion_reasons": exclusion_reasons,
        "min_text_length": min_text_length,
        "group_by": row_stats.get("group_by"),
        "notes": [
            "Empty-text rows are counted only as empty_or_missing_text, not also as invalid traits.",
            "missing_or_invalid_traits rows are not removed at ingestion; they remain grouped.",
            "Cleaning exclusions are exclusive: invalid content, then too-short, then duplicate.",
            "unmapped_after_clean is applied after the cleaner accepted the tweet.",
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pandora_comments(
    pandora_file: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_text_length: int = 3,
    group_by: GroupBy = "traits",
) -> list[PreparedUserComments]:
    """
    Full ingestion pipeline: load -> group by (proxy) user -> clean
    (existing DataCleaner) -> drop unusable comments -> persist -> return.

    This is the only entry point downstream code should use. Q-learning
    calls this (or consumes its cached output file) and never touches
    PANDORA or the filesystem directly.

    Parameters
    ----------
    pandora_file
        Path to the PANDORA parquet export.
    output_path
        Where prepared data is persisted as JSON.
    min_text_length
        Forwarded to DataCleaner; comments shorter than this after
        cleaning are dropped, same as the X pipeline.
    group_by
        "traits" (default): group comments by the (O, C, E, A, N) proxy
        user key. A trait tuple seen only once automatically becomes a
        group of one -- that's the built-in fallback when no real group
        exists, no separate mode is needed for it.
        "row": disable grouping entirely, one comment per unit.

    Returns
    -------
    list[PreparedUserComments]
        One entry per (proxy) user with their cleaned CleanedContent
        comments and Big Five / ptype traits attached.
    """
    rows = _load_raw_rows(pandora_file)
    grouped, row_stats = _group_rows(rows, group_by)

    cleaner = DataCleaner(min_text_length=min_text_length)
    prepared, clean_stats = _clean_all(grouped, cleaner)

    quality = _build_ingestion_quality(
        row_stats,
        clean_stats,
        source=str(pandora_file),
        min_text_length=min_text_length,
    )

    global _LAST_INGESTION_QUALITY
    _LAST_INGESTION_QUALITY = quality

    _save_prepared(prepared, output_path)
    quality_path = quality_sidecar_path(output_path)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    with quality_path.open("w", encoding="utf-8") as fh:
        json.dump(quality, fh, ensure_ascii=False, indent=2)
    logger.info("Saved PANDORA ingestion quality to %s", quality_path)

    logger.info(
        "PANDORA ingestion complete: %d users, %d total cleaned comments.",
        len(prepared),
        sum(len(u.comments) for u in prepared),
    )
    return prepared
  