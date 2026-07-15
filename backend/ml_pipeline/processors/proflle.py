"""
Profile Signal Processor
========================
Extracts personality-relevant features from a user's profile content object.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from ..cleaning.cleaner import CleanedContent
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class ProfileSignalProcessor(BaseProcessor):
    """
    Extracts features from a 'profile' CleanedContent item.

    Features
    --------
    bio_length          : Character length of cleaned bio
    bio_word_count      : Word count of cleaned bio
    bio_lexical_density : Unique words / total words (0 if no words)
    has_bio             : 1 if bio is non-empty, else 0
    bio_hashtag_count   : Number of hashtags in bio
    bio_mention_count   : Number of mentions in bio
    bio_emoji_count     : Number of emojis in bio
    bio_url_count       : Number of URLs stripped from bio
    verified            : 1 if account is verified, else 0
    followers_count     : Raw follower count
    following_count     : Raw following count
    follow_ratio        : followers / (following + 1) to avoid div-by-zero
    tweet_count         : Total tweet count from profile
    listed_count        : Times listed by others
    account_age_days    : Days since account creation (0 if unknown)
    has_location        : 1 if location field is set, else 0
    """

    def process(self, content: CleanedContent) -> dict[str, Any]:
        if content.content_type != "profile":
            logger.warning(
                "ProfileSignalProcessor received content_type=%r; expected 'profile'.",
                content.content_type,
            )

        bio = content.cleaned_text
        words = bio.split() if bio else []
        unique_words = set(words)

        meta = content.metadata
        followers = self._safe_int(meta.get("followers_count"))
        following = self._safe_int(meta.get("following_count"))
        verified = bool(meta.get("verified"))

        account_age_days = self._compute_account_age(meta.get("created_at"))

        return {
            "bio_length": len(bio),
            "bio_word_count": len(words),
            "bio_lexical_density": len(unique_words) / len(words) if words else 0.0,
            "has_bio": 1 if bio else 0,
            "bio_hashtag_count": len(content.signals.hashtags),
            "bio_mention_count": len(content.signals.mentions),
            "bio_emoji_count": len(content.signals.emojis),
            "bio_url_count": len(content.signals.urls),
            "verified": int(verified),
            "followers_count": followers,
            "following_count": following,
            "follow_ratio": followers / (following + 1),
            "tweet_count": self._safe_int(meta.get("tweet_count")),
            "listed_count": self._safe_int(meta.get("listed_count")),
            "account_age_days": account_age_days,
            "has_location": 1 if meta.get("location") else 0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_account_age(created_at: Any) -> float:
        """
        Return the age of the account in fractional days.
        Returns 0.0 if created_at cannot be parsed.
        """
        if not created_at:
            return 0.0
        try:
            if isinstance(created_at, (int, float)):
                dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(
                    str(created_at).replace("Z", "+00:00"))
            now = datetime.now(tz=timezone.utc)
            delta = now - dt
            return max(0.0, delta.total_seconds() / 86_400)
        except Exception as exc:
            logger.debug(
                "Could not parse account creation date %r: %s", created_at, exc)
            return 0.0
