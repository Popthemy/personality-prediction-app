"""
Twitter / X Post Fetching Service.

Uses x-tweet-fetcher (xtf) library to pull public posts from X without
requiring any API keys. Timelines are fetched via Nitter instances
configured in settings.XTF_NITTER.

Key responsibilities:
- Build an xtf.Router from settings
- Fetch a user's recent timeline
- Store posts in the POST model (deduplication via x_post_id)
- Parse Nitter's relative time strings into UTC datetimes
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from xtf.exceptions import XtfError

logger = logging.getLogger('ml_pipeline')


def _parse_time_ago(time_ago: str) -> datetime:
    """
    Convert Nitter's relative time string to an approximate UTC datetime.
    Nitter returns strings like "2h", "3d", "1 Jan 2025", "Jan 1, 2025".
    Falls back to utcnow() when the string cannot be parsed.
    """
    now = datetime.now(tz=timezone.utc)
    time_ago = (time_ago or '').strip()

    # Absolute date formats: "Jan 1, 2025" or "1 Jan 2025"
    for fmt in ('%b %d, %Y', '%d %b %Y', '%d %B %Y', '%B %d, %Y'):
        try:
            return datetime.strptime(time_ago, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Relative patterns: "5m", "2h", "3d"
    m = re.match(r'^(\d+)\s*([smhdw])', time_ago.lower())
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        deltas = {'s': timedelta(seconds=value),
                  'm': timedelta(minutes=value),
                  'h': timedelta(hours=value),
                  'd': timedelta(days=value),
                  'w': timedelta(weeks=value)}
        return now - deltas.get(unit, timedelta())

    logger.debug(f"Could not parse time_ago='{time_ago}', using current time")
    return now


class TwitterFetcher:
    """
    Service to fetch X posts for a volunteer using x-tweet-fetcher (xtf).

    Usage:
        fetcher = TwitterFetcher()
        saved, skipped = fetcher.fetch_and_save(volunteer)
    """

    # Module-level singleton to avoid repeated TCP/SSL setup
    _shared_router = None

    def __init__(self, nitter_instances=None):
        from django.conf import settings
        nitter_raw = nitter_instances or getattr(
            settings, 'XTF_NITTER',
            'https://nitter.kareem.one,https://nitter.privacyredirect.com,https://nitter.tiekoetter.com'
        )
        if isinstance(nitter_raw, str):
            self._nitter_list = [u.strip() for u in nitter_raw.split(',') if u.strip()]
        else:
            self._nitter_list = list(nitter_raw)

        self._max_posts = getattr(settings, 'XTF_MAX_POSTS', 50)

    def _get_router(self, force_new: bool = False, nitter_instances=None):
        """Return (or create) the module-level shared Router."""
        if force_new or TwitterFetcher._shared_router is None:
            from xtf import Router
            instances = nitter_instances if nitter_instances is not None else self._nitter_list
            TwitterFetcher._shared_router = Router(nitter_instances=instances)
            logger.info(f"[TwitterFetcher] Router created with {len(instances)} Nitter instances")
        return TwitterFetcher._shared_router

    def _fetch_profile_page_timeline(self, username: str, instance: str) -> List:
        """
        Fetch a timeline by reading the public Nitter profile page directly.

        Some accounts return no search results even though the profile page still
        renders posts. This path gives us a second chance without needing the
        browser backend to be available.
        """
        from xtf import http
        from xtf.models import Tweet
        from xtf.parsers.nitter_html import _extract_tweets_from_events, _parse_html

        base = instance.rstrip("/")
        url = f"{base}/{username}"
        html = http.get_text(url, timeout=15)
        raw_tweets = _extract_tweets_from_events(_parse_html(html).events)
        print(f"HTML tweets {html}")
        print(f"Raw tweets {raw_tweets}")
        tweets = [Tweet.from_nitter_entry(item) for item in raw_tweets]
        logger.info(
            f"[TwitterFetcher] direct profile fetch via {base} returned {len(tweets)} tweets for @{username}"
        )
        return tweets


    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fetch_timeline(self, x_handle: str) -> List:
        """Fetch recent tweets for *x_handle* and return raw xtf.Tweet list."""
        username = x_handle.lstrip('@')
        logger.info(f"[TwitterFetcher] fetching timeline for @{username}")
        try:
            router = self._get_router()
            tweets = router.fetch_timeline(username, limit=self._max_posts)

            # If the shared router returns nothing, retry each instance separately.
            if not tweets and len(self._nitter_list) > 1:
                from xtf import Router

                seen_ids = set()
                fallback_tweets = []
                for instance in self._nitter_list:
                    try:
                        alt_router = Router(nitter_instances=[instance])
                        alt_tweets = alt_router.fetch_timeline(username, limit=self._max_posts)
                        logger.info(
                            f"[TwitterFetcher] fallback via {instance} returned {len(alt_tweets)} tweets for @{username}"
                        )
                        for tweet in alt_tweets:
                            tweet_id = str(getattr(tweet, 'tweet_id', '') or '').strip()
                            if tweet_id and tweet_id not in seen_ids:
                                seen_ids.add(tweet_id)
                                fallback_tweets.append(tweet)
                    except Exception as instance_error:
                        logger.warning(
                            f"[TwitterFetcher] fallback instance {instance} failed for @{username}: {instance_error}"
                        )

                if fallback_tweets:
                    tweets = fallback_tweets[: self._max_posts]

            # Final fallback: read the public profile page directly.
            if not tweets and len(self._nitter_list) > 0:
                profile_tweets = []
                seen_ids = set()
                for instance in self._nitter_list:
                    try:
                        direct_tweets = self._fetch_profile_page_timeline(username, instance)
                        for tweet in direct_tweets:
                            tweet_id = str(getattr(tweet, 'tweet_id', '') or '').strip()
                            if not tweet_id or tweet_id in seen_ids:
                                continue
                            seen_ids.add(tweet_id)
                            profile_tweets.append(tweet)
                            if len(profile_tweets) >= self._max_posts:
                                break
                        if profile_tweets:
                            tweets = profile_tweets[: self._max_posts]
                            break
                    except XtfError as instance_error:
                        logger.warning(
                            f"[TwitterFetcher] direct profile fetch via {instance} failed for @{username}: {instance_error}"
                        )
                    except Exception as instance_error:
                        logger.warning(
                            f"[TwitterFetcher] direct profile fetch via {instance} errored for @{username}: {instance_error}"
                        )

            logger.info(f"[TwitterFetcher] got {len(tweets)} tweets for @{username}")
            return tweets
        except Exception as e:
            logger.error(f"[TwitterFetcher] fetch_timeline failed for @{username}: {e}")
            return []

    def fetch_and_save(self, volunteer) -> Tuple[int, int]:
        """
        Fetch posts for *volunteer* and save new ones to the POST DB table.

        Returns:
            (saved_count, skipped_count)  — skipped = already in DB
        """
        from backend.core.models import POST

        tweets = self.fetch_timeline(volunteer.x_handle)
        if not tweets:
            return 0, 0

        saved = 0
        skipped = 0

        for tweet in tweets:
            tweet_id = str(tweet.tweet_id or '').strip()
            if not tweet_id:
                skipped += 1
                continue

            # Deduplication
            if POST.objects.filter(x_post_id=tweet_id).exists():
                skipped += 1
                continue

            text = (tweet.text or '').strip()
            if not text:
                skipped += 1
                continue

            # Engagement
            likes = int(tweet.likes or 0)
            retweets = int(tweet.retweets or 0)
            replies = int(tweet.replies or 0)
            engagement = likes + retweets * 2 + replies

            # Detect reply / RT
            is_reply = text.startswith('@')
            is_retweet = (tweet.retweeted_by is not None) or text.startswith('RT @')

            # Date
            created_at = _parse_time_ago(tweet.time_ago)

            try:
                POST.objects.create(
                    volunteer=volunteer,
                    x_post_id=tweet_id,
                    content=text,
                    created_at_original=created_at,
                    like_count=likes,
                    retweet_count=retweets,
                    reply_count=replies,
                    engagement_score=float(engagement),
                    is_reply=is_reply,
                    is_retweet=is_retweet,
                    language_detected='en',
                )
                saved += 1
            except Exception as e:
                logger.error(f"[TwitterFetcher] failed to save tweet {tweet_id}: {e}")
                skipped += 1

        logger.info(
            f"[TwitterFetcher] @{volunteer.x_handle}: saved={saved}, skipped={skipped}"
        )
        return saved, skipped
