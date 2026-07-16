"""
Text Preprocessing Pipeline for social media posts.

Performs standard NLP pre-cleaning steps before embeddings:
1. URL removal  (t.co short links, full http(s) links)
2. Mention removal  (@username)
3. Hashtag normalisation (keep word, remove #)
4. Whitespace normalisation
5. Emoji / non-ASCII filtering (configurable)
6. Minimum length filtering

Usage:
    from backend.ml_pipeline.processors.text_preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    cleaned  = preprocessor.clean("Check this out! https://t.co/abc @user #AI")
    is_valid = preprocessor.is_valid(cleaned)
"""

import logging
import re
import unicodedata
from typing import List, Optional

logger = logging.getLogger('ml_pipeline')


class TextPreprocessor:
    """
    Social-media text cleaner tailored for BERT embedding input.

    Parameters
    ----------
    min_length : int
        Minimum character length for a post to be considered valid (post-cleaning).
    keep_emojis : bool
        When False (default), strip non-ASCII characters such as emoji.
    keep_hashtags : bool
        When True (default), keep the word behind '#' (removes the symbol only).
    """

    # Pre-compiled patterns for speed
    _URL_RE      = re.compile(r'http\S+|www\.\S+|t\.co/\S+', re.IGNORECASE)
    _MENTION_RE  = re.compile(r'@\w+')
    _HASHTAG_RE  = re.compile(r'#(\w+)')
    _RT_RE       = re.compile(r'^RT\s+', re.IGNORECASE)
    _SPACES_RE   = re.compile(r'\s+')
    _NON_ASCII   = re.compile(r'[^\x00-\x7F]+')

    def __init__(
        self,
        min_length: int = 15,
        keep_emojis: bool = False,
        keep_hashtags: bool = True,
    ):
        self.min_length   = min_length
        self.keep_emojis  = keep_emojis
        self.keep_hashtags = keep_hashtags

    # ------------------------------------------------------------------ #
    #  Core cleaning                                                       #
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on *text*."""
        if not text:
            return ''

        # 1. Strip leading RT prefix
        text = self._RT_RE.sub('', text)

        # 2. Remove URLs
        text = self._URL_RE.sub(' ', text)

        # 3. Remove @mentions
        text = self._MENTION_RE.sub('', text)

        # 3b. Strip leading colon/whitespace left after RT mention removal
        text = re.sub(r'^[:\s]+', '', text)

        # 4. Handle hashtags
        if self.keep_hashtags:
            text = self._HASHTAG_RE.sub(r'\1', text)
        else:
            text = self._HASHTAG_RE.sub('', text)

        # 5. Emoji / non-ASCII
        if not self.keep_emojis:
            text = self._NON_ASCII.sub(' ', text)

        # 6. Normalise whitespace (also cleans up double spaces from removed tokens)
        text = re.sub(r'\s*:\s*', ': ', text)  # normalise colons
        text = self._SPACES_RE.sub(' ', text).strip()
        # Remove leading/trailing punctuation left by removals
        text = re.sub(r'^[:\-,;!?.\s]+', '', text).strip()

        return text

    def is_valid(self, text: str) -> bool:
        """Return True if *text* passes minimum quality threshold."""
        return len(text) >= self.min_length

    # ------------------------------------------------------------------ #
    #  Batch helpers                                                       #
    # ------------------------------------------------------------------ #

    def process_posts(self, posts) -> List[dict]:
        """
        Clean and filter a queryset or list of POST model instances.

        Returns a list of dicts ready for embedding:
            [{'post_id': int, 'clean_text': str}, ...]
        """
        results = []
        for post in posts:
            raw_text = post.content or ''
            clean = self.clean(raw_text)
            if not self.is_valid(clean):
                logger.debug(
                    f"[TextPreprocessor] post {post.id} filtered out "
                    f"(length={len(clean)})"
                )
                continue
            results.append({'post_id': post.id, 'clean_text': clean})

        logger.info(
            f"[TextPreprocessor] {len(results)}/{len(list(posts))} posts passed filter"
            if hasattr(posts, '__len__')
            else f"[TextPreprocessor] processed {len(results)} posts"
        )
        return results

    def clean_batch(self, texts: List[str]) -> List[Optional[str]]:
        """
        Clean a plain list of strings.
        Returns None for entries that do not pass the validity check.
        """
        out = []
        for t in texts:
            c = self.clean(t)
            out.append(c if self.is_valid(c) else None)
        return out
