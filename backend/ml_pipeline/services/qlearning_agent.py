"""
Q-Learning Active Signal Selection for Comment Selection (PANDORA dataset)
===========================================================================
Implements tabular Q-learning (Sutton & Barto, 2018, Ch.6 §6.5) to learn
which *cleaned Reddit comments* are most worth handing to a downstream
BERT encoder for personality prediction on the PANDORA dataset.

WHAT CHANGED FROM THE ORIGINAL (ENGAGEMENT-BASED) DESIGN
----------------------------------------------------------
The original agent selected social-media *posts* using engagement /
recency / hashtag / URL features, and was rewarded by the drop in MAE
that a downstream BERT → Lasso predictor produced.

The PANDORA pipeline instead hands this agent a plain list of already
*cleaned* Reddit comment strings, e.g.:

    [
      "I finally finished my project and feel really proud of the result.",
      "I usually prefer working alone rather than in large groups.",
      "That movie was amazing, I would definitely watch it again.",
    ]

There is no engagement/recency metadata and no cheap way to query the
downstream BERT model at every RL step. So the redesign drops the
external MAE-based reward entirely and instead teaches the agent to do
what should happen *before* BERT ever sees the data: pick a compact,
information-dense, low-redundancy subset of a user's comments.

    • STATE   now describes the *current candidate comment* (length,
      lexical informativeness) together with *selection context*: how
      similar/redundant it is to comments already selected, and how
      much selection budget remains.
    • REWARD  now rewards informativeness and novelty (1 − redundancy)
      of a selected comment, minus a small per-selection cost, instead
      of a downstream-MAE improvement.
    • ACTIONS, the Q-table, the ε-greedy policy, and the Bellman update
      are unchanged.

WHAT IT IS NOT
--------------
• NOT a DQN            — the Q-function is a plain Python dict (tabular).
• NOT a contextual bandit — transitions are sequential; the state changes
  after every action, and the agent reasons about what it has already
  selected (redundancy is only meaningful relative to that history).
• NOT dependent on BERT/label feedback at training time — informativeness
  and redundancy are computed directly from comment text, so the agent
  can be trained cheaply and its output subset is *then* fed to BERT.

SUTTON & BARTO ALGORITHM (§6.5 Q-learning) — UNCHANGED
--------------------------------------------------------
    Initialise Q(s, a) arbitrarily for all s ∈ S, a ∈ A
    For each episode:
        S ← environment.reset()
        Loop for each step t of episode:
            A ← ε-greedy(S, Q)
            S', R, done ← environment.step(A)
            Q(S,A) ← Q(S,A) + α[R + γ · max_a Q(S',a) − Q(S,A)]
            S ← S'
        until done

MDP FORMALISATION (REDESIGNED)
--------------------------------
State  S_t  =  discretised tuple describing the candidate comment and
               the selection context:
    • length_bin              — char length of candidate comment  {short, medium, long}
    • diversity_bin           — lexical diversity (type-token ratio) of the
                                 candidate comment, a cheap informativeness
                                 proxy                                {low, medium, high}
    • redundancy_bin          — max text similarity between the candidate
                                 and every already-selected comment
                                 {none (nothing selected yet), low, medium, high}
    • n_selected_bin          — how many comments already chosen        {few, some, many}
    • n_remaining_bin         — how many candidates remain              {few, some, many}
    • budget_bin              — selection slots left (max_selected − n_selected)
                                                                          {tight, moderate, ample}
    • selection_ratio_bin     — selected / (selected + remaining)       {low, medium, high}
    • selected_diversity_bin  — mean lexical diversity of the already-selected
                                 set (composition summary, keeps the state
                                 closer to Markov — see Section 2)       {none, low, medium, high}

Action A_t ∈ {select, skip}

Reward R_{t+1}:
    select →  R = w_info · informativeness(comment)
                + w_novel · (1 − redundancy(comment, selected_so_far))
                − selection_cost
    skip   →  R = 0   (no cost, no benefit — mirrors the original design)

    informativeness(comment) ∈ [0, 1]  — lexical diversity (type-token ratio)
    redundancy(comment, selected) ∈ [0, 1] — max cosine similarity between the
        candidate's term-frequency vector and every already-selected
        comment's vector (0 when nothing has been selected yet)

    A comment that is both information-dense (high diversity) and
    dissimilar to everything already chosen (low redundancy) earns the
    highest reward — precisely the "informative and non-redundant"
    criterion requested for the BERT-bound subset.

Episode terminates when:
    • all candidate comments have been considered, OR
    • max_selected comments have been selected.

Pipeline integration
---------------------
The public API stays backward-compatible with the original module so
existing callers do not break:
    QLearningAgent          — identical constructor signature
    agent.choose_action()   — identical signature + return type
    agent.update_q_value()  — identical signature + return type
    agent.get_q_value()     — unchanged
    agent.save_state()      — identical return type
    agent.load_state()      — identical signature
    agent.select_posts()    — kept as an alias of the new select_comments()
    agent.featurize_post()  — kept as an alias of the new featurize_comment()
    create_post_features()  — kept as a compatibility bridge; now also
                               accepts a raw comment string directly

New public API for the PANDORA text pipeline (preferred going forward):
    create_comment_features()     — str → feature dict
    CommentSelectionEnvironment   — MDP environment over List[str] comments
    QLearningAgent.select_comments()
    QLearningAgent.featurize_comment()
    run_training_episode() / run_training_loop() — updated to the new MDP
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("ml_pipeline")


# ---------------------------------------------------------------------------
# SECTION 1 — Text feature & similarity helpers
# ---------------------------------------------------------------------------
# Everything here operates on the raw *cleaned* comment string. No external
# model call is required, which is what lets the agent be trained cheaply,
# purely from text, before the (expensive) BERT step ever runs.

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> List[str]:
    """Lightweight whitespace/punctuation tokeniser (lower-cased words)."""
    return _TOKEN_RE.findall((text or "").lower())


def _term_freq_vector(text: str) -> Dict[str, float]:
    """
    Default 'embedding_fn': a normalised bag-of-words term-frequency
    vector. This is deliberately cheap (no model call) so the agent can
    be trained/queried at every RL step without touching BERT.

    Pipelines that want *semantic* (not just lexical) redundancy detection
    can inject a real embedding function instead — see
    CommentSelectionEnvironment(embedding_fn=...) — e.g. cached
    sentence-BERT vectors turned into a {dim_index: value} dict, or any
    other representation that _cosine_similarity() can compare.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = sum(counts.values())
    return {word: count / total for word, count in counts.items()}


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    if not vec_a or not vec_b:
        return 0.0
    common_keys = set(vec_a) & set(vec_b)
    numerator = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _lexical_diversity(text: str) -> float:
    """
    Default 'informativeness_fn': type-token ratio (unique words / total
    words) of the comment. A cheap, model-free proxy for how information-
    dense a short comment is — repetitive or filler-heavy comments score
    low, comments with varied vocabulary score high.

    Pipelines that want a richer informativeness signal (e.g. entropy of
    a language-model's token distribution, or a saliency score) can
    inject informativeness_fn=... on CommentSelectionEnvironment.
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ---------------------------------------------------------------------------
# SECTION 1b — Discretisation ("binning") helpers
# ---------------------------------------------------------------------------
# All continuous/count features are bucketed before being stored in the
# Q-table so the state space remains finite and manageable (tabular).

def _bin_length(chars: int) -> str:
    """
    Discretise comment length in characters.

    >200 → 'long'   : detailed, substantive comment
    >50  → 'medium' : moderate length
    else → 'short'  : brief comment / near-empty
    """
    if chars > 200:
        return "long"
    if chars > 50:
        return "medium"
    return "short"


def _bin_diversity(ratio: float) -> str:
    """
    Discretise lexical diversity (type-token ratio) of a single comment.

    >=0.75 → 'high'   : highly varied vocabulary, information-dense
    >=0.45 → 'medium' : moderate variety
    else   → 'low'    : repetitive / low information content

    Short comments naturally trend toward high TTR (few chances to repeat
    a word), which is intentional here: brevity that still says something
    distinct is exactly what the agent should value.
    """
    if ratio >= 0.75:
        return "high"
    if ratio >= 0.45:
        return "medium"
    return "low"


def _bin_redundancy(similarity: Optional[float]) -> str:
    """
    Discretise the candidate comment's max similarity to the
    already-selected set.

    None    → 'none'   : nothing selected yet, redundancy undefined
    >=0.5   → 'high'   : near-duplicate of something already chosen
    >=0.2   → 'medium' : some topical/lexical overlap
    else    → 'low'    : largely novel relative to the selected set
    """
    if similarity is None:
        return "none"
    if similarity >= 0.5:
        return "high"
    if similarity >= 0.2:
        return "medium"
    return "low"


def _bin_count(n: int, low: int = 3, high: int = 8) -> str:
    """
    Discretise an integer count into three ordered buckets.

    Used for both *n_selected* and *n_remaining* so the agent can reason
    about how far into the episode it is without storing raw integers
    (which would explode the Q-table).
    """
    if n < low:
        return "few"
    if n < high:
        return "some"
    return "many"


def _bin_ratio(ratio: float) -> str:
    """
    Discretise selection ratio = n_selected / (n_selected + n_remaining).

    Gives the agent a coarse sense of how selective it has been so far,
    important for avoiding the degenerate policies of "select everything"
    or "skip everything".
    """
    if ratio < 0.25:
        return "low"
    if ratio < 0.60:
        return "medium"
    return "high"


def _bin_budget(slots_left: int, low: int = 2, high: int = 6) -> str:
    """
    Discretise the *selection budget* — how many more comments the agent
    is still allowed to pick (max_selected − n_selected).

    <=2  → 'tight'    : almost out of budget, be selective
    <=6  → 'moderate' : some room left
    else → 'ample'    : plenty of budget remaining

    This is distinct from n_remaining_bin (candidates left to look at):
    budget is about *how many more it may choose*, remaining is about
    *how many more it will get to see*.
    """
    if slots_left <= low:
        return "tight"
    if slots_left <= high:
        return "moderate"
    return "ample"


def _bin_selected_diversity(selected_diversities: List[float]) -> str:
    """
    Summarise the *informativeness composition* of the already-selected
    set (mean lexical diversity of selected comments).

    Markov-state note: the marginal value of the next comment depends on
    what kind of comments have already been chosen, not just how many.
    A set of already information-dense, distinct comments raises the bar
    for what counts as "worth adding"; a set of low-diversity comments
    means almost anything informative is a good addition.

    Returns
    -------
    "none"   — no comments selected yet
    "low" / "medium" / "high" — per _bin_diversity() on the mean
    """
    if not selected_diversities:
        return "none"
    mean_diversity = sum(selected_diversities) / len(selected_diversities)
    return _bin_diversity(mean_diversity)


# ---------------------------------------------------------------------------
# SECTION 2 — State dataclass
# ---------------------------------------------------------------------------

@dataclass
class CommentSelectionState:
    """
    Complete MDP state for one timestep of the comment-selection episode.

    This is what the Q-table indexes. Every field is a discrete string so
    the state can be hashed as a JSON key.

    Fields
    ------
    length_bin              : length tier of the *current candidate* comment
    diversity_bin            : lexical-diversity (informativeness) tier of
                                the current candidate comment
    redundancy_bin           : how similar the candidate is to the
                                already-selected set
    n_selected_bin            : coarse count of comments already chosen
    n_remaining_bin           : coarse count of comments still to consider
    budget_bin                : coarse selection-slots-remaining tier
    selection_ratio_bin      : selected / (selected + remaining), binned
    selected_diversity_bin   : mean informativeness tier of the selected set
    """
    length_bin:              str   # "short" | "medium" | "long"
    diversity_bin:           str   # "low" | "medium" | "high"
    redundancy_bin:          str   # "none" | "low" | "medium" | "high"
    n_selected_bin:          str   # "few" | "some" | "many"
    n_remaining_bin:         str   # "few" | "some" | "many"
    budget_bin:               str  # "tight" | "moderate" | "ample"
    selection_ratio_bin:     str   # "low" | "medium" | "high"
    selected_diversity_bin:  str   # "none" | "low" | "medium" | "high"

    def to_key(self) -> str:
        """
        Serialise state to a compact, sortable JSON string suitable as a
        Python dict key. This is the *state hash* used in the Q-table.
        """
        return json.dumps(
            {
                "len":   self.length_bin,
                "div":   self.diversity_bin,
                "redun": self.redundancy_bin,
                "nsel":  self.n_selected_bin,
                "nrem":  self.n_remaining_bin,
                "budget": self.budget_bin,
                "ratio": self.selection_ratio_bin,
                "sdiv":  self.selected_diversity_bin,
            },
            sort_keys=True,
        )


# Backward-compatible alias: code that still imports PostSelectionState
# (the pre-redesign name) keeps working unchanged.
PostSelectionState = CommentSelectionState


# ---------------------------------------------------------------------------
# SECTION 3 — MDP Environment
# ---------------------------------------------------------------------------

class CommentSelectionEnvironment:
    """
    Episodic MDP environment for sequential, informativeness/redundancy
    aware comment selection.

    One *episode* corresponds to processing all candidate comments for a
    single user (or a single batch of cleaned PANDORA comments). The agent
    steps through the comments one at a time, deciding to select or skip
    each one, so that the final selected subset is compact, informative,
    and non-redundant before it is handed to BERT.

    Reward (Sutton & Barto §3.2 — goals and rewards)
    -------------------------------------------------
        select →  R = w_info · informativeness(comment)
                    + w_novel · (1 − redundancy(comment, selected_so_far))
                    − selection_cost
        skip   →  R = 0   (no cost, no benefit)

    informativeness_fn and embedding_fn are both injectable:
        informativeness_fn(text: str) -> float in [0, 1]
        embedding_fn(text: str) -> Dict[str, float]   (used for redundancy
            via cosine similarity between candidate and selected vectors)

    Defaults are cheap, model-free text statistics (lexical diversity /
    term-frequency cosine similarity) so training never needs to call
    BERT. A pipeline that wants semantic redundancy detection can pass in
    real (cached) sentence embeddings instead.

    Episode termination (§3.3 — returns and episodes)
    --------------------------------------------------
    Episode ends when:
        1. All candidate comments have been considered, OR
        2. n_selected reaches max_selected.
    """

    ACTIONS = ("select", "skip")

    def __init__(
        self,
        comments: Sequence[Union[str, Dict[str, Any]]],
        embedding_fn: Optional[Callable[[str], Dict[str, float]]] = None,
        informativeness_fn: Optional[Callable[[str], float]] = None,
        selection_cost: float = 0.05,
        max_selected: int = 10,
        informativeness_weight: float = 1.0,
        novelty_weight: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        comments : Ordered list of cleaned Reddit comments. Each element
                    may be a raw string (the common case — see the module
                    docstring's example input) or a pre-built feature dict
                    from create_comment_features(). Strings are converted
                    automatically.
        embedding_fn : Callable[[str], Dict[str, float]] — turns a comment
                    into a sparse vector used to measure redundancy against
                    the selected set via cosine similarity. Defaults to a
                    normalised bag-of-words term-frequency vector.
        informativeness_fn : Callable[[str], float] — informativeness score
                    in [0, 1] for a comment. Defaults to lexical diversity
                    (type-token ratio).
        selection_cost : Per-selection penalty subtracted from reward.
                    Prevents the degenerate "always select" policy.
                    Recommended range: 0.01 – 0.10.
        max_selected  : Hard cap on selections per episode — this is the
                    *selection budget* reflected in state via budget_bin.
        informativeness_weight, novelty_weight : Relative weight given to
                    "the comment itself is informative" vs "the comment is
                    unlike anything already chosen" in the reward. Equal
                    weighting (1.0 / 1.0) is a reasonable default; raise
                    novelty_weight to push harder against redundancy.
        """
        self._embedding_fn = embedding_fn or _term_freq_vector
        self._informativeness_fn = informativeness_fn or _lexical_diversity
        self._selection_cost = selection_cost
        self._max_selected = max_selected
        self._informativeness_weight = informativeness_weight
        self._novelty_weight = novelty_weight

        self._comments: List[Dict[str, Any]] = [
            c if isinstance(c, dict) else create_comment_features(c, comment_id=i)
            for i, c in enumerate(comments)
        ]

        # Episode state — reset() initialises these properly.
        self._cursor:               int                    = 0
        self._selected:             List[Dict[str, Any]]   = []
        self._selected_vectors:     List[Dict[str, float]] = []
        self._selected_diversities: List[float]             = []
        self._step_count:           int                    = 0
        self._done:                 bool                   = False

    # ------------------------------------------------------------------
    # Public MDP interface (Sutton & Barto §3.6 finite MDPs)
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """
        Reset the environment for a new episode. Unlike the original
        MAE-based design, there is no external baseline to query here —
        informativeness/redundancy are computed directly from text at
        each step, so reset() only needs to clear episode bookkeeping.

        Returns
        -------
        state_key : str — JSON-serialised initial state (Q-table key)
        """
        self._cursor = 0
        self._selected = []
        self._selected_vectors = []
        self._selected_diversities = []
        self._step_count = 0
        self._done = False

        logger.debug(
            "Environment reset: %d candidate comments | max_selected=%d",
            len(self._comments),
            self._max_selected,
        )
        return self._observe()

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Apply *action* to the current candidate comment.

        This is the core MDP transition function:
            (S_t, A_t) → (S_{t+1}, R_{t+1}, done)

        Parameters
        ----------
        action : 'select' or 'skip'

        Returns
        -------
        next_state : str  — Q-table key for S_{t+1}
        reward     : float
        done       : bool
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() first.")

        current = self._comments[self._cursor]
        text = current.get("text", "")

        if action == "select":
            vector = self._embedding_fn(text)
            informativeness = self._informativeness_fn(text)
            redundancy = self._max_similarity(vector)

            reward = self._compute_reward(informativeness, redundancy)

            self._selected.append(current)
            self._selected_vectors.append(vector)
            self._selected_diversities.append(informativeness)
        else:
            # skip: zero reward — no cost, no benefit
            reward = 0.0

        self._cursor += 1
        self._step_count += 1

        self._done = (
            self._cursor >= len(self._comments)
            or len(self._selected) >= self._max_selected
        )

        next_state = self._observe()

        logger.debug(
            "Step %d | action=%s | reward=%.4f | done=%s | n_sel=%d",
            self._step_count,
            action,
            reward,
            self._done,
            len(self._selected),
        )

        return next_state, reward, self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_similarity(self, vector: Dict[str, float]) -> Optional[float]:
        """
        Max cosine similarity between *vector* and every already-selected
        comment's vector. None if nothing has been selected yet (redundancy
        is undefined against an empty set).
        """
        if not self._selected_vectors:
            return None
        return max(_cosine_similarity(vector, v) for v in self._selected_vectors)

    def _compute_reward(self, informativeness: float, redundancy: Optional[float]) -> float:
        """
        Compute reward for a 'select' action.

        R = w_info · informativeness + w_novel · (1 − redundancy) − selection_cost

        • High informativeness, low redundancy → large positive reward
          (exactly what we want to feed BERT).
        • Near-duplicate of an already-selected comment → redundancy → 1,
          novelty term → 0, reward collapses toward −selection_cost.
        • Nothing selected yet → redundancy defaults to 0 (no prior
          duplicate is possible), so the first pick is judged purely on
          its own informativeness.
        """
        redundancy_value = redundancy if redundancy is not None else 0.0
        novelty = 1.0 - redundancy_value
        return (
            self._informativeness_weight * informativeness
            + self._novelty_weight * novelty
            - self._selection_cost
        )

    def _observe(self) -> str:
        """
        Build the current MDP state and return its Q-table key.

        Terminal state
        --------------
        When the episode is finished (cursor past all comments, or budget
        exhausted) we return a dedicated terminal-state key using neutral
        bins for the (nonexistent) current candidate, while keeping the
        selected-set summary (selected_diversity_bin) accurate. Its
        Q-values remain 0 by convention (no future reward after
        termination).
        """
        n_selected = len(self._selected)
        n_remaining = max(len(self._comments) - self._cursor, 0)
        total = n_selected + n_remaining
        ratio = n_selected / total if total > 0 else 0.0
        slots_left = max(self._max_selected - n_selected, 0)

        selected_diversity_bin = _bin_selected_diversity(self._selected_diversities)

        if self._done or self._cursor >= len(self._comments):
            return CommentSelectionState(
                length_bin="short",
                diversity_bin="low",
                redundancy_bin="none",
                n_selected_bin=_bin_count(n_selected),
                n_remaining_bin="few",
                budget_bin=_bin_budget(slots_left),
                selection_ratio_bin=_bin_ratio(ratio),
                selected_diversity_bin=selected_diversity_bin,
            ).to_key()

        comment = self._comments[self._cursor]
        text = comment.get("text", "")
        vector = self._embedding_fn(text)
        redundancy = self._max_similarity(vector)

        return CommentSelectionState(
            length_bin=_bin_length(comment.get("text_length", len(text))),
            diversity_bin=_bin_diversity(self._informativeness_fn(text)),
            redundancy_bin=_bin_redundancy(redundancy),
            n_selected_bin=_bin_count(n_selected),
            n_remaining_bin=_bin_count(n_remaining),
            budget_bin=_bin_budget(slots_left),
            selection_ratio_bin=_bin_ratio(ratio),
            selected_diversity_bin=selected_diversity_bin,
        ).to_key()

    # ------------------------------------------------------------------
    # Read-only accessors (useful for the training loop)
    # ------------------------------------------------------------------

    @property
    def selected_comments(self) -> List[Dict[str, Any]]:
        return list(self._selected)

    @property
    def selected_diversities(self) -> List[float]:
        return list(self._selected_diversities)

    @property
    def n_candidates(self) -> int:
        return len(self._comments)

    @property
    def is_done(self) -> bool:
        return self._done


# Backward-compatible alias: code that still imports/refers to
# PostSelectionEnvironment (the pre-redesign name) keeps working.
PostSelectionEnvironment = CommentSelectionEnvironment


# ---------------------------------------------------------------------------
# SECTION 4 — Tabular Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning agent for active, redundancy-aware comment
    selection.

    Implements the off-policy TD control algorithm from
    Sutton & Barto (2018), Chapter 6, Section 6.5 — UNCHANGED by this
    redesign:

        Q(S_t, A_t) ← Q(S_t, A_t)
                       + α [ R_{t+1} + γ · max_a Q(S_{t+1}, a) − Q(S_t, A_t) ]

    WHAT IS STORED IN THE Q-TABLE
    ------------------------------
    The Q-table maps (state_key, action) → expected cumulative reward.
    A high Q(s, 'select') means: "given the current comment's own
    informativeness and its similarity to what's already been selected,
    plus how much selection budget remains, choosing this comment is
    expected to build the most useful subset for BERT."

    EXPLORATION vs EXPLOITATION (§2.4, §6.5) — UNCHANGED
    ------------------------------------------------------
    ε-greedy: with probability ε the agent picks an action at random
    (exploration); otherwise it picks the action with the highest Q-value
    (exploitation). ε is typically decayed over training.

    API COMPATIBILITY
    -----------------
    choose_action, update_q_value, get_q_value, save_state, load_state
    are unchanged. select_posts / featurize_post are kept as aliases of
    the new select_comments / featurize_comment so existing pipeline call
    sites keep working unchanged.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ) -> None:
        """
        Initialise Q-Learning agent.

        Parameters
        ----------
        alpha   : Learning rate α ∈ (0, 1].
                  How quickly new information overrides old estimates.
        gamma   : Discount factor γ ∈ [0, 1).
                  How much the agent values future reward vs immediate
                  reward. γ close to 1 → the agent plans ahead, which
                  matters here because an early skip preserves budget
                  for a later, less-redundant comment.
        epsilon : Exploration rate ε ∈ [0, 1]. Set to 0 at evaluation time.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: state_key (str) → {action (str) → Q-value (float)}
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.actions = list(CommentSelectionEnvironment.ACTIONS)  # ['select', 'skip']

        logger.info(
            "QLearningAgent initialised | α=%.3f | γ=%.3f | ε=%.3f",
            alpha, gamma, epsilon,
        )

    # ------------------------------------------------------------------
    # State featurisation
    # ------------------------------------------------------------------

    def featurize_comment(self, comment_data: Union[str, Dict[str, Any]]) -> str:
        """
        Convert a single comment (raw string or feature dict) to a
        Q-table state key, using neutral defaults for episode-context
        fields (no episode is running).

        DO use for: logging, quick per-comment diagnostics.
        DO NOT use for: actual inference / selection decisions — for
        that use select_comments(), which runs the comment sequence
        through CommentSelectionEnvironment and produces consistent
        episode-context states (redundancy and budget only mean
        something in the context of a running episode).

        Parameters
        ----------
        comment_data : a cleaned comment string, or a dict produced by
                        create_comment_features().

        Returns
        -------
        JSON string — Q-table key (same schema as CommentSelectionState.to_key())
        """
        if isinstance(comment_data, str):
            comment_data = create_comment_features(comment_data)
        text = comment_data.get("text", "")

        return CommentSelectionState(
            length_bin=_bin_length(comment_data.get("text_length", len(text))),
            diversity_bin=_bin_diversity(comment_data.get("lexical_diversity", _lexical_diversity(text))),
            redundancy_bin="none",
            n_selected_bin="few",
            n_remaining_bin="some",
            budget_bin="ample",
            selection_ratio_bin="low",
            selected_diversity_bin="none",
        ).to_key()

    def featurize_post(self, post_data: Union[str, Dict[str, Any]]) -> str:
        """Backward-compatible alias for featurize_comment()."""
        return self.featurize_comment(post_data)

    # ------------------------------------------------------------------
    # Q-table accessors — unchanged
    # ------------------------------------------------------------------

    def get_q_value(self, state: str, action: str) -> float:
        """Return Q(state, action). Unseen (s, a) pairs initialise to 0.0."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state].get(action, 0.0)

    def _ensure_state(self, state: str) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    # ------------------------------------------------------------------
    # Action selection — ε-greedy (Sutton & Barto §2.4) — unchanged
    # ------------------------------------------------------------------

    def choose_action(self, state: str, training: bool = True) -> str:
        """
        Select an action using ε-greedy policy.

        During *training*: with probability ε → random action (explore);
        with probability 1−ε → highest-Q action (exploit).
        During *evaluation* (training=False): always greedy, ties broken
        randomly.

        Parameters
        ----------
        state    : Q-table key (output of featurize_comment or env._observe())
        training : If False, acts purely greedily (ε=0).

        Returns
        -------
        'select' or 'skip'
        """
        if training and np.random.random() < self.epsilon:
            chosen = np.random.choice(self.actions)
            logger.debug("ε-greedy EXPLORE → %s", chosen)
            return chosen

        self._ensure_state(state)
        q_vals = self.q_table[state]
        max_q = max(q_vals.values())
        best = [a for a, v in q_vals.items() if v == max_q]
        chosen = np.random.choice(best)
        logger.debug("ε-greedy EXPLOIT → %s (Q=%.4f)", chosen, max_q)
        return chosen

    # ------------------------------------------------------------------
    # Q-Learning update (Sutton & Barto §6.5, eq. 6.8) — unchanged
    # ------------------------------------------------------------------

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
    ) -> Tuple[float, float]:
        """
        Apply the tabular Q-learning update rule:

            Q(S_t, A_t) ← Q(S_t, A_t)
                           + α [R_{t+1} + γ · max_a Q(S_{t+1}, a) − Q(S_t, A_t)]

        Off-policy TD(0): the max over next-state actions makes the target
        the greedy (optimal) estimate regardless of which action is
        actually taken next — this is what separates Q-learning from Sarsa.

        Returns
        -------
        (old_q_value, new_q_value)
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())

        td_error = reward + self.gamma * max_next_q - old_q
        new_q = old_q + self.alpha * td_error

        self.q_table[state][action] = new_q

        logger.debug(
            "Q-update | s=%s… | a=%s | R=%.4f | Q: %.4f → %.4f | δ=%.4f",
            state[:40], action, reward, old_q, new_q, td_error,
        )

        return old_q, new_q

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def select_comments(
        self,
        comments: Sequence[Union[str, Dict[str, Any]]],
        top_k: int = 10,
        training: bool = False,
        embedding_fn: Optional[Callable[[str], Dict[str, float]]] = None,
        informativeness_fn: Optional[Callable[[str], float]] = None,
        selection_cost: float = 0.05,
        informativeness_weight: float = 1.0,
        novelty_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Select up to *top_k* comments using the trained Q-policy.

        Inference steps through the *same* CommentSelectionEnvironment
        used during training, so every state observed at inference time
        is built by exactly the same _observe() logic and is therefore
        queried against Q-table keys the agent actually saw in training
        (train/inference state consistency).

        No Q-learning updates are performed (training=False → greedy
        policy, no Bellman update called).

        HOW THE TOP-K LIMIT WORKS
        --------------------------
        top_k is enforced via the environment's max_selected parameter
        (it also shapes budget_bin in the state). The agent may select
        fewer than top_k comments if the greedy policy judges remaining
        candidates too redundant/uninformative (Q(s, 'skip') > Q(s, 'select')).

        Parameters
        ----------
        comments      : List of cleaned comment strings (or feature dicts
                         from create_comment_features()) — the candidate
                         pool for one user/episode.
        top_k          : Maximum number of comments to return.
        training       : If True, uses ε-greedy; if False (default), acts greedily.
        embedding_fn, informativeness_fn, selection_cost,
        informativeness_weight, novelty_weight : forwarded to the
                         environment; must match the values used during
                         training for state/reward consistency.

        Returns
        -------
        List of selected comment dicts, each augmented with:
            q_value : float — Q(state, 'select') at the moment of selection
            state   : str   — full Q-table key (including episode context)
            action  : str   — always 'select'
        Ordered by selection sequence, not by Q-value (episode context
        changes after each selection, so Q-values aren't directly
        comparable across steps).
        """
        env = CommentSelectionEnvironment(
            comments=comments,
            embedding_fn=embedding_fn,
            informativeness_fn=informativeness_fn,
            selection_cost=selection_cost,
            max_selected=top_k,
            informativeness_weight=informativeness_weight,
            novelty_weight=novelty_weight,
        )

        state = env.reset()
        selected: List[Dict[str, Any]] = []

        while not env.is_done:
            action = self.choose_action(state, training=training)
            q_val = self.get_q_value(state, "select")

            next_state, _reward, done = env.step(action)

            if action == "select":
                chosen_comment = env.selected_comments[-1]
                selected.append(
                    {
                        **chosen_comment,
                        "q_value": q_val,
                        "state": state,
                        "action": "select",
                    }
                )

            state = next_state

        logger.info(
            "select_comments | selected %d / %d | top_k=%d",
            len(selected), len(comments), top_k,
        )
        return selected

    def select_posts(
        self,
        posts: Sequence[Union[str, Dict[str, Any]]],
        top_k: int = 10,
        training: bool = False,
        predictor_fn=None,
        selection_cost: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible alias for select_comments().

        predictor_fn is accepted for signature compatibility with the
        original engagement/MAE-based pipeline but is no longer used —
        the redesigned reward is computed directly from comment text
        (informativeness + novelty), not from a downstream predictor.
        """
        if predictor_fn is not None:
            logger.warning(
                "select_posts(): predictor_fn is deprecated and ignored "
                "under the text-based informativeness/redundancy reward; "
                "pass embedding_fn/informativeness_fn to select_comments() instead."
            )
        return self.select_comments(
            posts, top_k=top_k, training=training, selection_cost=selection_cost,
        )

    # ------------------------------------------------------------------
    # Persistence — unchanged
    # ------------------------------------------------------------------

    def save_state(self) -> Dict[str, Any]:
        """Serialise the full agent state to a plain dict (JSON-safe)."""
        return {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore agent state from a previously serialised dict."""
        self.q_table = {k: dict(v) for k, v in state_dict.get("q_table", {}).items()}
        self.alpha = state_dict.get("alpha", self.alpha)
        self.gamma = state_dict.get("gamma", self.gamma)
        self.epsilon = state_dict.get("epsilon", self.epsilon)
        logger.info(
            "QLearningAgent state loaded | Q-table size=%d states",
            len(self.q_table),
        )


# ---------------------------------------------------------------------------
# SECTION 5 — Training loop
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Diagnostic statistics recorded for one training episode."""
    episode:              int
    total_reward:         float
    n_selected:           int
    n_candidates:         int
    mean_informativeness: float
    mean_redundancy:      float
    n_q_updates:          int
    mean_td_error:        float


def run_training_episode(
    agent: QLearningAgent,
    env: CommentSelectionEnvironment,
    episode: int = 0,
) -> EpisodeStats:
    """
    Execute one complete Q-learning training episode.

    Implements the inner loop from Sutton & Barto §6.5:

        S ← env.reset()
        while not done:
            A  ← ε-greedy(S, Q)
            S', R, done ← env.step(A)
            Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) − Q(S,A)]
            S ← S'

    Parameters
    ----------
    agent   : QLearningAgent — updated in-place
    env     : CommentSelectionEnvironment — reset at the start
    episode : Episode index (for logging only)

    Returns
    -------
    EpisodeStats with per-episode diagnostics
    """
    state = env.reset()

    total_reward = 0.0
    td_errors: List[float] = []
    done = False

    while not done:
        action = agent.choose_action(state, training=True)
        next_state, reward, done = env.step(action)

        # Update for both select and skip so the agent learns the value
        # of restraint (skip → reward=0) as well as selection.
        old_q, new_q = agent.update_q_value(state, action, reward, next_state)
        td_errors.append(abs(new_q - old_q))

        total_reward += reward
        state = next_state

    selected_diversities = env.selected_diversities
    mean_informativeness = float(np.mean(selected_diversities)) if selected_diversities else 0.0

    stats = EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        n_selected=len(env.selected_comments),
        n_candidates=env.n_candidates,
        mean_informativeness=mean_informativeness,
        mean_redundancy=0.0,  # populated below once selections exist
        n_q_updates=len(td_errors),
        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
    )

    logger.info(
        "Episode %4d | R=%.4f | selected=%d/%d | mean_info=%.4f | "
        "Q-updates=%d | mean|δ|=%.4f | Q-table=%d states",
        episode, stats.total_reward, stats.n_selected, stats.n_candidates,
        stats.mean_informativeness, stats.n_q_updates, stats.mean_td_error,
        len(agent.q_table),
    )

    return stats


def run_training_loop(
    agent: QLearningAgent,
    comment_batches: List[List[Union[str, Dict[str, Any]]]],
    n_epochs: int = 5,
    selection_cost: float = 0.05,
    max_selected: int = 10,
    informativeness_weight: float = 1.0,
    novelty_weight: float = 1.0,
    embedding_fn: Optional[Callable[[str], Dict[str, float]]] = None,
    informativeness_fn: Optional[Callable[[str], float]] = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    predictor_fn=None,
) -> List[EpisodeStats]:
    """
    Full Q-learning training loop (Sutton & Barto §6.5).

    WHAT THIS LOOP LEARNS
    ---------------------
    The agent iterates over every user's/document's comment collection
    (*comment_batches*, one inner list of cleaned comment strings per
    episode) for *n_epochs* epochs. Reward comes directly from comment
    text (informativeness + novelty vs the selected set), so no
    downstream model call is needed during training.

    After training, agent.select_comments() embodies the learned policy:
    it picks comments whose state has high Q(s, 'select') — i.e. comments
    the agent has learned are worth spending selection budget on given
    their own informativeness and their redundancy against what's already
    chosen.

    ε-DECAY SCHEDULE — unchanged
    ------------------------------
        ε ← max(ε_end, ε · decay)   after every episode

    Parameters
    ----------
    agent                    : QLearningAgent — updated in-place
    comment_batches          : List of per-user/-document comment lists
                                 (each is one episode)
    n_epochs                 : How many full passes over all batches
    selection_cost            : Per-selection penalty forwarded to the environment
    max_selected              : Selection budget per episode
    informativeness_weight,
    novelty_weight             : Reward weighting forwarded to the environment
    embedding_fn, informativeness_fn : Optional overrides forwarded to the
                                 environment (e.g. real BERT embeddings)
    epsilon_start/end/decay   : ε-greedy annealing schedule
    predictor_fn               : Deprecated, accepted only for backward
                                 compatibility; ignored under the new reward.

    Returns
    -------
    List[EpisodeStats] — one entry per episode (epoch × n_batches)
    """
    if predictor_fn is not None:
        logger.warning(
            "run_training_loop(): predictor_fn is deprecated and ignored "
            "under the text-based informativeness/redundancy reward."
        )

    agent.epsilon = epsilon_start
    all_stats: List[EpisodeStats] = []
    episode_idx = 0

    for epoch in range(n_epochs):
        epoch_reward = 0.0

        for comments in comment_batches:
            env = CommentSelectionEnvironment(
                comments=comments,
                embedding_fn=embedding_fn,
                informativeness_fn=informativeness_fn,
                selection_cost=selection_cost,
                max_selected=max_selected,
                informativeness_weight=informativeness_weight,
                novelty_weight=novelty_weight,
            )

            stats = run_training_episode(agent, env, episode=episode_idx)
            all_stats.append(stats)
            epoch_reward += stats.total_reward
            episode_idx += 1

            agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        logger.info(
            "Epoch %d/%d complete | mean episode reward=%.4f | ε=%.4f | Q-table=%d",
            epoch + 1, n_epochs, epoch_reward / max(len(comment_batches), 1),
            agent.epsilon, len(agent.q_table),
        )

    logger.info(
        "Training complete | total episodes=%d | final Q-table=%d states",
        episode_idx, len(agent.q_table),
    )
    return all_stats


# ---------------------------------------------------------------------------
# SECTION 6 — Pipeline integration helpers (backward-compatible)
# ---------------------------------------------------------------------------

def create_comment_features(text: str, comment_id: Optional[Any] = None) -> Dict[str, Any]:
    """
    Convert a single *cleaned* Reddit comment string into the feature
    dict consumed by featurize_comment() / CommentSelectionEnvironment().

    This is the primary entry point for the PANDORA text pipeline: the
    dataset hands the agent a list of cleaned comment strings, and each
    one is converted with this function.

    Parameters
    ----------
    text        : cleaned comment string
    comment_id  : optional identifier (e.g. Reddit comment id or list
                  index) carried through to the selected-comment output
                  of select_comments() for traceability.

    Returns
    -------
    Dict with keys:
        comment_id        : Any (as passed in)
        text               : str (the original cleaned comment)
        text_length        : int  (characters)
        lexical_diversity  : float  (type-token ratio, in [0, 1])
        tf_vector           : Dict[str, float]  (term-frequency vector,
                               used by the default redundancy check)
    """
    return {
        "comment_id": comment_id,
        "text": text,
        "text_length": len(text or ""),
        "lexical_diversity": _lexical_diversity(text),
        "tf_vector": _term_freq_vector(text),
    }


def create_post_features(post_obj_or_text: Union[str, Any], comment_id: Optional[Any] = None) -> Dict[str, Any]:
    """
    Backward-compatible bridge to the original create_post_features().

    Accepts either:
      - a raw / cleaned comment string  (current PANDORA text pipeline), or
      - a legacy object exposing .cleaned_content / .content (the old
        Django Post-style engagement pipeline).

    In both cases the result is the create_comment_features() dict — the
    engagement/recency/hashtag/URL fields from the original engagement-
    based pipeline are not produced, since the redesigned state no longer
    uses them.
    """
    if isinstance(post_obj_or_text, str):
        return create_comment_features(post_obj_or_text, comment_id=comment_id)

    text = getattr(post_obj_or_text, "cleaned_content", None) or getattr(post_obj_or_text, "content", "")
    resolved_id = getattr(post_obj_or_text, "id", comment_id)
    return create_comment_features(text, comment_id=resolved_id)
