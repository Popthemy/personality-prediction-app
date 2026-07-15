"""
Network Signal Processor
=========================
Extracts network-level features from the user profile.
Designed for extensibility: graph-based metrics (PageRank, clustering
coefficient, community membership) can be added later by subclassing
or injecting a NetworkGraphProvider.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ..cleaning.cleaner import CleanedContent
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class NetworkSignalProcessor(BaseProcessor):
    """
    Extracts network/social-graph features.

    Current features (derived from profile metadata only):
    -------------------------------------------------------
    followers_count         : Raw followers
    following_count         : Raw following
    follow_ratio            : followers / (following + 1)
    log_followers           : log(1 + followers)
    log_following           : log(1 + following)
    network_reach_score     : Normalised proxy for broadcast reach [0, 1]
    is_influencer           : 1 if followers > 10 000, else 0
    is_broadcaster          : 1 if follow_ratio > 10, else 0

    Extensibility hooks (always 0 until graph data is injected):
    ------------------------------------------------------------
    graph_pagerank          : Placeholder for future PageRank score
    graph_clustering_coeff  : Placeholder for clustering coefficient
    graph_community_id      : Placeholder for community label
    """

    # Threshold above which an account is considered an influencer
    INFLUENCER_THRESHOLD = 10_000

    def __init__(self, graph_provider: Any = None) -> None:
        """
        Parameters
        ----------
        graph_provider
            Optional future object implementing `.get_graph_features(user_id)`
            that returns a dict with graph_pagerank, graph_clustering_coeff,
            and graph_community_id. Pass None to use placeholder zeros.
        """
        self._graph_provider = graph_provider

    def process(self, content: CleanedContent) -> dict[str, Any]:
        meta = content.metadata

        followers = self._safe_int(meta.get("followers_count"))
        following = self._safe_int(meta.get("following_count"))
        follow_ratio = followers / (following + 1)

        # Normalised reach: log-scaled followers capped at log(1M) = 13.8
        log_followers = math.log1p(followers)
        reach_max = math.log1p(1_000_000)
        network_reach = min(log_followers / reach_max, 1.0)

        # Graph features (extensibility hook)
        graph_features = self._get_graph_features(meta)

        return {
            "followers_count": followers,
            "following_count": following,
            "follow_ratio": round(follow_ratio, 4),
            "log_followers": round(log_followers, 4),
            "log_following": round(math.log1p(following), 4),
            "network_reach_score": round(network_reach, 6),
            "is_influencer": int(followers >= self.INFLUENCER_THRESHOLD),
            "is_broadcaster": int(follow_ratio > 10),
            **graph_features,
        }

    # ------------------------------------------------------------------
    # Extensibility hook
    # ------------------------------------------------------------------

    def _get_graph_features(self, meta: dict[str, Any]) -> dict[str, Any]:
        """
        Retrieve graph-based metrics from an optional graph provider.
        Falls back to zero-filled placeholders if no provider is configured.
        """
        placeholders: dict[str, Any] = {
            "graph_pagerank": 0.0,
            "graph_clustering_coeff": 0.0,
            "graph_community_id": -1,
        }
        if self._graph_provider is None:
            return placeholders

        user_id = meta.get("screen_name") or meta.get("id")
        try:
            graph_data = self._graph_provider.get_graph_features(user_id)
            placeholders.update(
                {k: graph_data[k] for k in placeholders if k in graph_data})
        except Exception as exc:
            logger.warning(
                "Graph provider failed for user %r: %s", user_id, exc)

        return placeholders
