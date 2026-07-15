"""
Engagement Signal Processor
============================
Extracts normalised engagement metrics from content metadata.
Log-scaling prevents high-viral outliers from dominating the feature space.
"""

from __future__ import annotations

import math
import logging
from typing import Any

from personality_pipeline.cleaning.cleaner import CleanedContent
from .base import BaseProcessor

logger = logging.getLogger(__name__)


def _log1p(x: float) -> float:
    """log(1 + x) — safe for x = 0."""
    return math.log1p(max(0.0, x))


class EngagementSignalProcessor(BaseProcessor):
    """
    Extracts engagement features from a CleanedContent item's metadata.

    Features
    --------
    like_count              : Raw like count
    reply_count             : Raw reply count
    repost_count            : Raw repost count
    quote_count             : Raw quote count
    total_engagement        : Sum of all four counts
    log_like_count          : log(1 + likes)
    log_reply_count         : log(1 + replies)
    log_repost_count        : log(1 + reposts)
    log_total_engagement    : log(1 + total)
    reply_ratio             : replies / (total + 1)
    repost_ratio            : reposts / (total + 1)
    virality_score          : (reposts + quotes) / (total + 1)
    """

    def process(self, content: CleanedContent) -> dict[str, Any]:
        meta = content.metadata
        likes = self._safe_int(meta.get("like_count"))
        replies = self._safe_int(meta.get("reply_count"))
        reposts = self._safe_int(meta.get("repost_count"))
        quotes = self._safe_int(meta.get("quote_count"))

        total = likes + replies + reposts + quotes

        return {
            "like_count": likes,
            "reply_count": replies,
            "repost_count": reposts,
            "quote_count": quotes,
            "total_engagement": total,
            "log_like_count": round(_log1p(likes), 4),
            "log_reply_count": round(_log1p(replies), 4),
            "log_repost_count": round(_log1p(reposts), 4),
            "log_total_engagement": round(_log1p(total), 4),
            "reply_ratio": round(replies / (total + 1), 4),
            "repost_ratio": round(reposts / (total + 1), 4),
            "virality_score": round((reposts + quotes) / (total + 1), 4),
        }
