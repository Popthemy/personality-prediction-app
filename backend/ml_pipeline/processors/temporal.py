"""
Temporal Signal Processor
==========================
Extracts time-based behavioural signals from content timestamps.
Recency scoring uses exponential decay (half-life = 30 days).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from personality_pipeline.cleaning.cleaner import CleanedContent
from .base import BaseProcessor

logger = logging.getLogger(__name__)

_HALF_LIFE_DAYS = 30.0  # recency decay half-life


class TemporalSignalProcessor(BaseProcessor):
    """
    Extracts temporal features from a CleanedContent item's timestamp.

    Features
    --------
    posting_hour            : Hour of day (0–23), -1 if unknown
    day_of_week             : 0=Monday … 6=Sunday, -1 if unknown
    is_weekend              : 1 if Sat/Sun, else 0
    recency_score           : Exponential decay; 1.0 = today, → 0 as age → ∞
    is_morning              : 1 if posting_hour in [6, 12)
    is_afternoon            : 1 if posting_hour in [12, 18)
    is_evening              : 1 if posting_hour in [18, 22)
    is_night                : 1 if posting_hour in [22, 24) or [0, 6)
    has_timestamp           : 1 if a valid timestamp was present, else 0
    """

    def __init__(self, reference_time: datetime | None = None) -> None:
        """
        Parameters
        ----------
        reference_time
            Used as "now" for recency calculation. Defaults to UTC now.
            Inject a fixed value in tests for determinism.
        """
        self._reference_time = reference_time or datetime.now(tz=timezone.utc)

    def process(self, content: CleanedContent) -> dict[str, Any]:
        ts = content.timestamp_utc
        if not ts:
            return self._unknown()

        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            logger.debug("Unparseable timestamp %r: %s", ts, exc)
            return self._unknown()

        hour = dt.hour
        dow = dt.weekday()
        is_weekend = int(dow >= 5)

        age_days = (self._reference_time - dt).total_seconds() / 86_400
        age_days = max(0.0, age_days)
        recency = math.exp(-math.log(2) * age_days / _HALF_LIFE_DAYS)

        return {
            "posting_hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "recency_score": round(recency, 6),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 18),
            "is_evening": int(18 <= hour < 22),
            "is_night": int(hour >= 22 or hour < 6),
            "has_timestamp": 1,
        }

    @staticmethod
    def _unknown() -> dict[str, Any]:
        return {
            "posting_hour": -1,
            "day_of_week": -1,
            "is_weekend": 0,
            "recency_score": 0.0,
            "is_morning": 0,
            "is_afternoon": 0,
            "is_evening": 0,
            "is_night": 0,
            "has_timestamp": 0,
        }
