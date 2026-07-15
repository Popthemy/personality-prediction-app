"""
Phase 1: Data Cleaning Layer
============================
Cleans raw X (Twitter) API data into normalized content objects
ready for downstream signal extraction.

Design principles:
- Non-destructive: original_text is always preserved alongside cleaned_text
- Safe: all field access is guarded against missing/null data
- Deterministic: same input always yields same output
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------

@dataclass
class RawXData:
    """Typed wrapper around raw X API payload."""
    profile: dict[str, Any] = field(default_factory=dict)
    tweets: list[dict[str, Any]] = field(default_factory=list)
    replies: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawXData":
        return cls(
            profile=data.get("profile") or {},
            tweets=data.get("tweets") or [],
            replies=data.get("replies") or [],
        )


@dataclass
class ExtractedSignals:
    """
    Inline extracted signals preserved during cleaning.
    These are metadata pulled *out* of the text before normalization
    so downstream processors can count them without re-parsing.
    """
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    emojis: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)


@dataclass
class CleanedContent:
    """
    A single normalized content item (tweet, reply, or profile bio).

    Fields
    ------
    content_id      : Stable hash derived from source content
    content_type    : "tweet" | "reply" | "profile"
    original_text   : Raw text exactly as received from the API
    cleaned_text    : Normalized text (URLs stripped, whitespace collapsed, lowercased)
    signals         : Pre-extracted structural signals (hashtags, mentions, etc.)
    timestamp_utc   : ISO-8601 UTC string, or None if unavailable
    metadata        : All remaining API fields (engagement counts, etc.)
    """
    content_id: str
    content_type: str
    original_text: str
    cleaned_text: str
    signals: ExtractedSignals
    timestamp_utc: str | None
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches http/https URLs
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Unicode emoji detection (covers most Emoji_Presentation + Emoji_Modifier ranges)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE,
)

_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def _stable_id(content_type: str, raw_id: Any, text: str) -> str:
    """
    Generate a stable, reproducible content ID.

    Priority: use the API-supplied id_str if present; otherwise hash the text.
    """
    if raw_id:
        return f"{content_type}_{raw_id}"
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{content_type}_{digest}"


def _parse_timestamp(raw: Any) -> str | None:
    """
    Normalize a timestamp to ISO-8601 UTC.

    Accepts:
    - Unix epoch (int / float)
    - ISO-8601 string with or without timezone
    - X API v1 format: "Mon Jan 01 00:00:00 +0000 2024"
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            ts = datetime.fromtimestamp(raw, tz=timezone.utc)
            return ts.isoformat()
        if isinstance(raw, str):
            raw = raw.strip()
            # X API v1 legacy format
            for fmt in (
                "%a %b %d %H:%M:%S %z %Y",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(raw, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc).isoformat()
                except ValueError:
                    continue
    except Exception as exc:  # pragma: no cover
        logger.debug("Timestamp parse failed for %r: %s", raw, exc)
    return None


def _extract_signals(text: str) -> tuple[ExtractedSignals, str]:
    """
    Extract structural signals from raw text and return them alongside
    a cleaned version with URLs removed (hashtags / mentions / emojis kept).
    """
    urls = _URL_RE.findall(text)
    hashtags = _HASHTAG_RE.findall(text)
    mentions = _MENTION_RE.findall(text)
    emojis = _EMOJI_RE.findall(text)

    # Remove URLs only (hashtags/mentions/emojis stay)
    cleaned = _URL_RE.sub("", text)
    # Collapse whitespace
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    # Lowercase
    cleaned = cleaned.lower()

    signals = ExtractedSignals(
        hashtags=[h.lower() for h in hashtags],
        mentions=[m.lower() for m in mentions],
        emojis=emojis,
        urls=urls,
    )
    return signals, cleaned


def _safe_str(value: Any, field_name: str = "unknown") -> str:
    """Coerce to string safely, logging unexpected types."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    logger.debug("Field %r has unexpected type %s; coercing.", field_name, type(value).__name__)
    return str(value)


# ---------------------------------------------------------------------------
# Main Cleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Cleans a raw RawXData payload into a list of CleanedContent objects.

    Usage
    -----
    >>> cleaner = DataCleaner()
    >>> results = cleaner.clean(RawXData.from_dict(raw_api_payload))
    >>> for item in results:
    ...     print(item.content_id, item.cleaned_text[:60])
    """

    def __init__(self, min_text_length: int = 3) -> None:
        """
        Parameters
        ----------
        min_text_length
            After cleaning, items whose cleaned_text is shorter than this
            (excluding the profile) are dropped as uninformative.
        """
        self.min_text_length = min_text_length
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, raw: RawXData) -> list[CleanedContent]:
        """
        Clean all content in a RawXData payload.

        Deduplication is performed across the entire payload: if two tweets
        (or a tweet and a reply) share identical cleaned_text they are
        collapsed to one item. The profile is never deduplicated.

        Returns
        -------
        list[CleanedContent]
            Ordered: profile first, then tweets, then replies.
        """
        self._seen_hashes = set()
        results: list[CleanedContent] = []

        # Profile (always included; never filtered by length or dedup)
        profile_item = self._clean_profile(raw.profile)
        if profile_item:
            results.append(profile_item)

        # Tweets
        for idx, tweet in enumerate(raw.tweets):
            item = self._clean_post(tweet, "tweet", idx)
            if item and self._accept(item):
                results.append(item)

        # Replies
        for idx, reply in enumerate(raw.replies):
            item = self._clean_post(reply, "reply", idx)
            if item and self._accept(item):
                results.append(item)

        logger.info(
            "Cleaned %d items from raw payload (%d tweets, %d replies).",
            len(results),
            len(raw.tweets),
            len(raw.replies),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accept(self, item: CleanedContent) -> bool:
        """
        Filter gate applied to every non-profile item.
        Returns False (drops item) if:
        - cleaned_text is too short
        - cleaned_text is a duplicate of a previously seen item
        """
        if len(item.cleaned_text) < self.min_text_length:
            logger.debug("Dropping %s (too short).", item.content_id)
            return False

        fingerprint = hashlib.md5(item.cleaned_text.encode()).hexdigest()
        if fingerprint in self._seen_hashes:
            logger.debug("Dropping %s (duplicate).", item.content_id)
            return False

        self._seen_hashes.add(fingerprint)
        return True

    def _clean_profile(self, profile: dict[str, Any]) -> CleanedContent | None:
        if not profile:
            logger.warning("Profile field is empty or missing.")
            return None

        raw_bio = _safe_str(profile.get("description") or profile.get("bio"), "bio")
        signals, cleaned_bio = _extract_signals(raw_bio)

        # Build a clean metadata dict with all numeric / boolean profile fields
        metadata = {
            "verified": bool(profile.get("verified") or profile.get("is_verified")),
            "followers_count": int(profile.get("followers_count") or 0),
            "following_count": int(profile.get("friends_count") or profile.get("following_count") or 0),
            "tweet_count": int(profile.get("statuses_count") or profile.get("tweet_count") or 0),
            "listed_count": int(profile.get("listed_count") or 0),
            "created_at": profile.get("created_at"),
            "location": _safe_str(profile.get("location"), "location"),
            "name": _safe_str(profile.get("name"), "name"),
            "screen_name": _safe_str(profile.get("screen_name") or profile.get("username"), "screen_name"),
        }

        return CleanedContent(
            content_id=_stable_id("profile", profile.get("id_str") or profile.get("id"), raw_bio),
            content_type="profile",
            original_text=raw_bio,
            cleaned_text=cleaned_bio,
            signals=signals,
            timestamp_utc=_parse_timestamp(profile.get("created_at")),
            metadata=metadata,
        )

    def _clean_post(
        self,
        post: dict[str, Any],
        content_type: str,
        fallback_idx: int,
    ) -> CleanedContent | None:
        if not post:
            return None

        # Support both v1 ("text"/"full_text") and v2 ("text") field names
        raw_text = _safe_str(
            post.get("full_text") or post.get("text") or post.get("body"),
            "text",
        )
        signals, cleaned = _extract_signals(raw_text)

        raw_id = post.get("id_str") or post.get("id") or fallback_idx
        timestamp = _parse_timestamp(
            post.get("created_at") or post.get("timestamp") or post.get("timestamp_ms")
        )

        metadata = {
            "like_count": int(post.get("favorite_count") or post.get("like_count") or 0),
            "reply_count": int(post.get("reply_count") or 0),
            "repost_count": int(post.get("retweet_count") or post.get("repost_count") or 0),
            "quote_count": int(post.get("quote_count") or 0),
            "in_reply_to_user_id": post.get("in_reply_to_user_id_str") or post.get("in_reply_to_user_id"),
            "lang": post.get("lang"),
            "source": post.get("source"),
        }

        return CleanedContent(
            content_id=_stable_id(content_type, raw_id, raw_text),
            content_type=content_type,
            original_text=raw_text,
            cleaned_text=cleaned,
            signals=signals,
            timestamp_utc=timestamp,
            metadata=metadata,
        )