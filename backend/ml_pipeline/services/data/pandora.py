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

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = Path("backend/ml_pipeline/data/prepared/pandora_prepared.json")

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
) -> dict[str, tuple[UserTraits | None, list[str]]]:
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

    for i, row in enumerate(rows):
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
            logger.debug("Row %d missing/invalid trait columns; storing without traits.", i)

        if key not in grouped:
            grouped[key] = (traits, [])
        grouped[key][1].append(str(text))

    if skipped:
        logger.warning("Skipped %d PANDORA rows with empty/missing text.", skipped)

    n_singletons = sum(1 for _, texts in grouped.values() if len(texts) == 1)
    logger.info(
        "Grouped %d rows into %d proxy users (%d are singleton groups).",
        skipped + sum(len(t) for _, t in grouped.values()),
        len(grouped),
        n_singletons,
    )
    return grouped


# ---------------------------------------------------------------------------
# Cleaning (delegated to existing DataCleaner) + regrouping by user
# ---------------------------------------------------------------------------

def _clean_all(
    grouped: dict[str, tuple[UserTraits | None, list[str]]],
    cleaner: DataCleaner,
) -> list[PreparedUserComments]:
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
    for item in cleaned_items:
        if item.content_type != "tweet":
            continue  # skip the (absent) profile item; we only fed tweets
        suffix = item.content_id.rsplit("_p", 1)
        if len(suffix) != 2 or not suffix[1].isdigit():
            logger.debug("Could not map cleaned item %s back to a user; dropping.", item.content_id)
            continue
        idx = int(suffix[1])
        user_id = idx_to_user.get(idx)
        if user_id is not None:
            grouped_prepared[user_id].append(item)

    return [
        PreparedUserComments(
            user_id=user_id,
            traits=grouped[user_id][0],
            comments=items,
        )
        for user_id, items in grouped_prepared.items()
        if items  # drop users left with zero usable comments after cleaning
    ]


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
    grouped = _group_rows(rows, group_by)

    cleaner = DataCleaner(min_text_length=min_text_length)
    prepared = _clean_all(grouped, cleaner)

    _save_prepared(prepared, output_path)

    logger.info(
        "PANDORA ingestion complete: %d users, %d total cleaned comments.",
        len(prepared),
        sum(len(u.comments) for u in prepared),
    )
    return prepared