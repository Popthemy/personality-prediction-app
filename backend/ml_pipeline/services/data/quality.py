"""
Measured data-quality helpers for the PANDORA experiment path.

This module does not clean, filter, or split data. It only formats counts
that callers already measured, with explicit denominators and safe rates.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence


def safe_rate(
    numerator: Optional[int],
    denominator: Optional[int],
    *,
    denominator_name: str,
) -> Dict[str, Any]:
    """Return a rate that can be recalculated; percent is None if denominator is 0."""
    num = None if numerator is None else int(numerator)
    den = None if denominator is None else int(denominator)
    percent = None
    if num is not None and den is not None and den > 0:
        percent = round(100.0 * num / den, 4)
    return {
        "numerator": num,
        "denominator": den,
        "denominator_name": denominator_name,
        "percent": percent,
    }


def exclusion(count: Optional[int], reason: str, *, stage: str) -> Dict[str, Any]:
    return {
        "count": None if count is None else int(count),
        "reason": reason,
        "stage": stage,
    }


def comment_volume(
    counts: Sequence[int],
    *,
    split: str,
    population: str,
) -> Dict[str, Any]:
    """Per-participant comment counts for one disjoint population."""
    values = [int(c) for c in counts]
    n = len(values)
    total = int(sum(values)) if values else 0
    return {
        "split": split,
        "population": population,
        "n_participants": n,
        "n_comments": total,
        "mean_comments_per_participant": (
            round(total / n, 4) if n > 0 else None
        ),
        "median_comments_per_participant": (
            float(median(values)) if n > 0 else None
        ),
        "min_comments": min(values) if n > 0 else None,
        "max_comments": max(values) if n > 0 else None,
        "denominator": "participants in this population",
        "denominator_n": n,
    }


def thesis_rows(data_quality: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten headline counts into rows suitable for a thesis preprocessing table."""
    rows: List[Dict[str, Any]] = []

    def add(
        metric: str,
        count: Any,
        *,
        stage: str,
        denominator: Any = None,
        denominator_name: Optional[str] = None,
        percent: Any = None,
        reason: Optional[str] = None,
    ) -> None:
        rows.append({
            "stage": stage,
            "metric": metric,
            "count": count,
            "denominator": denominator,
            "denominator_name": denominator_name,
            "percent": percent,
            "reason": reason,
        })

    ingestion = data_quality.get("ingestion") or {}
    if ingestion.get("available"):
        add("raw_rows", ingestion.get("raw_rows"), stage="ingestion")
        empty = ingestion.get("empty_or_missing_text") or {}
        add(
            "empty_or_missing_text", empty.get("count"),
            stage="ingestion",
            denominator=ingestion.get("raw_rows"),
            denominator_name="raw_rows",
            reason=empty.get("reason"),
        )
        invalid = ingestion.get("missing_or_invalid_traits") or {}
        add(
            "missing_or_invalid_traits", invalid.get("count"),
            stage="ingestion",
            reason=invalid.get("reason"),
        )
        add(
            "rows_retained_for_grouping",
            ingestion.get("rows_retained_for_grouping"),
            stage="ingestion",
            denominator=ingestion.get("raw_rows"),
            denominator_name="raw_rows",
            percent=(ingestion.get("row_retention") or {}).get("percent"),
        )

    cleaning = data_quality.get("cleaning") or {}
    if cleaning.get("available"):
        add("comments_before_cleaning", cleaning.get("comments_before_cleaning"), stage="cleaning")
        for key in ("excluded_too_short", "excluded_duplicate", "excluded_invalid_content"):
            block = cleaning.get(key) or {}
            add(
                key, block.get("count"),
                stage="cleaning",
                denominator=cleaning.get("comments_before_cleaning"),
                denominator_name="comments_before_cleaning",
                reason=block.get("reason"),
            )
        add(
            "comments_retained",
            cleaning.get("comments_retained"),
            stage="cleaning",
            denominator=cleaning.get("comments_before_cleaning"),
            denominator_name="comments_before_cleaning",
            percent=(cleaning.get("comment_retention") or {}).get("percent"),
        )

    filt = data_quality.get("experiment_filter") or {}
    add("users_before_filter", filt.get("users_before_filter"), stage="experiment_filter")
    add("users_after_filter", filt.get("users_after_filter"), stage="experiment_filter")
    add("users_used", filt.get("users_used"), stage="experiment_filter")
    add("users_excluded", filt.get("users_excluded"), stage="experiment_filter")
    for key, block in (filt.get("exclusion_reasons") or {}).items():
        if isinstance(block, dict):
            add(
                f"users_excluded_{key}",
                block.get("count"),
                stage="experiment_filter",
                denominator=filt.get("users_before_filter"),
                denominator_name="users_before_filter",
                reason=block.get("reason"),
            )

    for name, vol in (data_quality.get("comment_volume") or {}).items():
        if not isinstance(vol, dict):
            continue
        add(
            f"{name}_n_participants",
            vol.get("n_participants"),
            stage="comment_volume",
            reason=vol.get("population"),
        )
        add(f"{name}_n_comments", vol.get("n_comments"), stage="comment_volume")
        add(
            f"{name}_mean_comments",
            vol.get("mean_comments_per_participant"),
            stage="comment_volume",
        )
        add(
            f"{name}_median_comments",
            vol.get("median_comments_per_participant"),
            stage="comment_volume",
        )
        add(f"{name}_min_comments", vol.get("min_comments"), stage="comment_volume")
        add(f"{name}_max_comments", vol.get("max_comments"), stage="comment_volume")

    extra: Iterable[Dict[str, Any]] = data_quality.get("thesis_rows") or []
    if extra and not rows:
        return list(extra)
    return rows
