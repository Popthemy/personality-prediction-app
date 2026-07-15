"""
Linguistic Signal Processor
============================
Extracts psycholinguistic and surface-level text features from tweets / replies.
Sentiment is computed with VADER (lexicon-based, no internet required).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..cleaning.cleaner import CleanedContent
from .base import BaseProcessor

logger = logging.getLogger(__name__)

# Compiled once at import time
_FIRST_PERSON_RE = re.compile(
    r"\b(i|me|my|mine|myself|i'm|i've|i'll|i'd)\b", re.IGNORECASE
)
_QUESTION_END_RE = re.compile(r"\?")
_EXCLAMATION_END_RE = re.compile(r"!")

# Shared VADER analyser (thread-safe read, not modified after init)
_VADER = SentimentIntensityAnalyzer()


class LinguisticSignalProcessor(BaseProcessor):
    """
    Extracts 11 linguistic / psycholinguistic features per content item.

    Features
    --------
    word_count              : Total word count of cleaned text
    char_count              : Character count (no spaces)
    lexical_diversity       : Unique lemmas / total words
    sentiment_compound      : VADER compound score [-1, 1]
    sentiment_positive      : VADER positive ratio
    sentiment_negative      : VADER negative ratio
    sentiment_neutral       : VADER neutral ratio
    emotion_intensity       : |compound| — how extreme the sentiment is
    emoji_count             : Number of emojis (from pre-extracted signals)
    hashtag_count           : Number of hashtags
    mention_count           : Number of @mentions
    first_person_ratio      : First-person token count / word count
    question_ratio          : ? count / sentence count (proxy)
    exclamation_ratio       : ! count / sentence count (proxy)
    """

    def process(self, content: CleanedContent) -> dict[str, Any]:
        text = content.cleaned_text
        original = content.original_text  # use original for VADER — it handles punctuation
        signals = content.signals

        words = text.split()
        word_count = len(words)
        char_count = len(text.replace(" ", ""))

        unique_words = set(words)
        lexical_diversity = len(unique_words) / \
            word_count if word_count else 0.0

        # VADER on the original (preserves punctuation for accuracy)
        vs = _VADER.polarity_scores(original)

        # Count sentence-ending punctuation as proxy sentence count
        sentence_count = max(
            1,
            len(re.findall(r"[.!?]", original))
        )

        first_person_tokens = len(_FIRST_PERSON_RE.findall(text))
        question_count = len(_QUESTION_END_RE.findall(original))
        exclamation_count = len(_EXCLAMATION_END_RE.findall(original))

        return {
            "word_count": word_count,
            "char_count": char_count,
            "lexical_diversity": round(lexical_diversity, 4),
            "sentiment_compound": round(vs["compound"], 4),
            "sentiment_positive": round(vs["pos"], 4),
            "sentiment_negative": round(vs["neg"], 4),
            "sentiment_neutral": round(vs["neu"], 4),
            "emotion_intensity": round(abs(vs["compound"]), 4),
            "emoji_count": len(signals.emojis),
            "hashtag_count": len(signals.hashtags),
            "mention_count": len(signals.mentions),
            "first_person_ratio": round(first_person_tokens / word_count, 4) if word_count else 0.0,
            "question_ratio": round(question_count / sentence_count, 4),
            "exclamation_ratio": round(exclamation_count / sentence_count, 4),
        }
