"""
Q-Learning Active Signal Selection for Post Selection
=====================================================
Implements tabular Q-learning (Sutton & Barto, 2018, Ch.6 §6.5) to learn
which social-media posts are *most informative for downstream Big Five
personality prediction*.

WHAT THE AGENT IS LEARNING
---------------------------
The agent learns a policy π: S → A that selects a compact, high-quality
subset of a user's posts so that, when those posts are fed through
  posts → BERT → Lasso/ElasticNet → Big Five (OCEAN) scores,
the prediction error (MAE over the five traits) is as low as possible.

It learns this purely from the reward signal:
    R_{t+1} = (MAE_before_step − MAE_after_step) − selection_cost

Positive reward → the post that was just selected reduced prediction error.
Negative reward → the post hurt performance or the agent selected too many.

The engagement score, recency, text length, etc. are *state features*,
not rewards.  The agent discovers which combinations of those features
predict downstream utility.

WHAT IT IS NOT
--------------
• NOT a DQN  — the Q-function is a plain Python dict (tabular).
• NOT a contextual bandit  — transitions are sequential; the state changes
  after every action; the agent reasons about what has already been selected.
• NOT trained on BFI-44 labels as input features  — labels only appear inside
  the environment to compute reward (no data leakage).

SUTTON & BARTO ALGORITHM (§6.5 Q-learning)
-------------------------------------------
    Initialise Q(s, a) arbitrarily for all s ∈ S, a ∈ A
    For each episode:
        S ← environment.reset()
        Loop for each step t of episode:
            A ← ε-greedy(S, Q)
            S', R, done ← environment.step(A)
            Q(S,A) ← Q(S,A) + α[R + γ · max_a Q(S',a) − Q(S,A)]
            S ← S'
        until done

MDP FORMALISATION
-----------------
State  S_t  =  discretised tuple of:
    • engagement bin of current candidate post  {low, medium, high}
    • recency bin                               {recent, medium, old}
    • text-length bin                           {short, medium, long}
    • has_hashtags                              {0, 1}
    • has_urls                                  {0, 1}
    • n_selected_bin  — how many posts already chosen  {few, some, many}
    • n_remaining_bin — how many candidates remain      {few, some, many}
    • selection_ratio_bin — selected / (selected+remaining) {low, med, high}
    • mae_bin — last known downstream MAE bucket  {none, high, medium, low}

Action A_t ∈ {select, skip}

Reward R_{t+1} = (mae_before − mae_after) − selection_cost
    where mae is computed by the pipeline's downstream predictor.
    If no downstream MAE is available yet, reward = −selection_cost  (skip is
    costless; the agent must earn reward by improving prediction quality).

Episode terminates when:
    • all candidate posts have been considered, OR
    • max_selected posts have been selected, OR
    • n_remaining == 0

Django integration
------------------
The public API is backward-compatible with the original module:
    QLearningAgent          — identical constructor signature
    agent.featurize_post()  — identical return type (JSON string)
    agent.choose_action()   — identical return type ('select'/'skip')
    agent.update_q_value()  — identical signature + return type
    agent.select_posts()    — identical signature + return type
    agent.save_state()      — identical return type
    agent.load_state()      — identical signature
    create_post_features()  — identical signature + return type

New public additions (do not break existing callers):
    PostSelectionEnvironment   — MDP environment
    run_training_episode()     — single-episode training function
    run_training_loop()        — full training loop
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ml_pipeline")


# ---------------------------------------------------------------------------
# SECTION 1 — State discretisation helpers
# ---------------------------------------------------------------------------
# All continuous/count features are bucketed before being stored in the
# Q-table so the state space remains finite and manageable (tabular).
#
# Bin boundaries are chosen to reflect realistic social-media distributions
# and are intentionally coarse to encourage generalisation.

def _bin_engagement(score: float) -> str:
    """
    Discretise engagement score.

    Engagement = likes + 2·retweets + 3·replies (normalised).
    >50  → 'high'   : viral / highly engaged post
    >10  → 'medium' : moderate engagement
    else → 'low'    : minimal or no engagement
    """
    if score > 50:
        return "high"
    if score > 10:
        return "medium"
    return "low"


def _bin_recency(days: float) -> str:
    """
    Discretise post age in days.

    <7   → 'recent' : within the last week
    <30  → 'medium' : within the last month
    else → 'old'    : older than a month
    """
    if days < 7:
        return "recent"
    if days < 30:
        return "medium"
    return "old"


def _bin_length(chars: int) -> str:
    """
    Discretise post text length in characters.

    >200 → 'long'   : detailed, substantive post
    >50  → 'medium' : moderate length
    else → 'short'  : brief post / near-empty
    """
    if chars > 200:
        return "long"
    if chars > 50:
        return "medium"
    return "short"


def _bin_count(n: int, low: int = 3, high: int = 8) -> str:
    """
    Discretise an integer count into three ordered buckets.

    Used for both *n_selected* and *n_remaining* so the agent can
    reason about how far into the episode it is without storing raw
    integers (which would explode the Q-table).
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
    which is important for avoiding the degenerate policies of
    "always select everything" or "always skip everything".
    """
    if ratio < 0.25:
        return "low"
    if ratio < 0.60:
        return "medium"
    return "high"


def _bin_mae(mae: Optional[float]) -> str:
    """
    Discretise the most-recently-observed downstream MAE.

    None / not-yet-computed → 'none'
    MAE > 1.0  → 'high'   : large prediction error; room for improvement
    MAE > 0.5  → 'medium' : moderate error
    else       → 'low'    : good prediction quality; harder to improve

    MAE is measured on the Big Five scale (raw Likert scores; range ≈ 1–5),
    so values outside [0, 4] are practically impossible.
    """
    if mae is None:
        return "none"
    if mae > 1.0:
        return "high"
    if mae > 0.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# SECTION 2 — State dataclass
# ---------------------------------------------------------------------------

def _bin_selected_engagement(selected: List[Dict]) -> str:
    """
    Summarise the *engagement distribution* of the already-selected set.

    FIX (Issue 4 — Markov state): the value of the next post depends not
    just on *how many* posts are selected but on *what kind* they are.
    Two posts with high engagement contribute differently than two posts
    with low engagement.  This bin lets the Q-table distinguish those
    histories without enumerating raw post IDs (which would explode the
    state space).

    Returns
    -------
    "none"   — no posts selected yet
    "low"    — mean engagement of selected set ≤ 10
    "medium" — mean engagement 10–50
    "high"   — mean engagement > 50
    """
    if not selected:
        return "none"
    mean_eng = sum(p.get("engagement_score", 0)
                   for p in selected) / len(selected)
    return _bin_engagement(mean_eng)


def _bin_selected_length(selected: List[Dict]) -> str:
    """
    Summarise the *text-length distribution* of the already-selected set.

    FIX (Issue 4 — Markov state): a selected set of long, detailed posts
    has different marginal-value dynamics than one of short posts.  Adding
    another long post to an already long-heavy set is less novel than adding
    it to a short-heavy set.  This bin captures that without raw counts.

    Returns
    -------
    "none"   — no posts selected yet
    "short"  — mean length ≤ 50 chars
    "medium" — mean length 50–200 chars
    "long"   — mean length > 200 chars
    """
    if not selected:
        return "none"
    mean_len = sum(p.get("text_length", 0) for p in selected) / len(selected)
    return _bin_length(int(mean_len))


@dataclass
class PostSelectionState:
    """
    Complete MDP state for one timestep of the post-selection episode.

    This is what the Q-table indexes.  Every field is a discrete string
    so the state can be hashed as a JSON key.

    MARKOV PROPERTY (Issue 4 fix)
    ------------------------------
    The original state only encoded *how many* posts were selected.  That
    is NOT Markov: the future value of a candidate post depends on the
    *composition* of the selected set, not merely its size.  Two agents
    that have each selected 3 posts may have very different selected sets
    and therefore very different optimal next actions.

    We add two compact *selected-set summary* fields:
        selected_engagement_bin — mean engagement tier of selected set
        selected_length_bin     — mean text-length tier of selected set

    These give the Q-table enough context to distinguish, e.g.,:
        "3 high-engagement long posts already chosen → next post adds diversity"
    from:
        "3 low-engagement short posts chosen → next long post adds a lot"

    This does not eliminate the Markov approximation entirely (tabular RL
    on discrete bins is always an approximation of the true continuous MDP),
    but it makes the approximation substantially tighter.

    Fields
    ------
    engagement_bin          : quality signal for the *current candidate* post
    recency_bin             : how fresh the current candidate is
    length_bin              : text-length of the current candidate
    has_hashtags            : whether the candidate contains hashtags
    has_urls                : whether the candidate contains URLs
    n_selected_bin          : coarse count of posts already chosen
    n_remaining_bin         : coarse count of posts still to be considered
    selection_ratio_bin     : selected / (selected + remaining), binned
    mae_bin                 : most-recently-observed downstream MAE, binned
    selected_engagement_bin : mean engagement tier of the selected set
    selected_length_bin     : mean text-length tier of the selected set
    """
    engagement_bin:          str   # "low" | "medium" | "high"
    recency_bin:             str   # "recent" | "medium" | "old"
    length_bin:              str   # "short" | "medium" | "long"
    has_hashtags:            bool
    has_urls:                bool
    n_selected_bin:          str   # "few" | "some" | "many"
    n_remaining_bin:         str   # "few" | "some" | "many"
    selection_ratio_bin:     str   # "low" | "medium" | "high"
    mae_bin:                 str   # "none" | "high" | "medium" | "low"
    selected_engagement_bin: str   # "none" | "low" | "medium" | "high"
    selected_length_bin:     str   # "none" | "short" | "medium" | "long"

    def to_key(self) -> str:
        """
        Serialise state to a compact, sortable JSON string suitable as a
        Python dict key.  This is the *state hash* used in the Q-table.
        """
        return json.dumps(
            {
                "eng":    self.engagement_bin,
                "rec":    self.recency_bin,
                "len":    self.length_bin,
                "htag":   int(self.has_hashtags),
                "url":    int(self.has_urls),
                "nsel":   self.n_selected_bin,
                "nrem":   self.n_remaining_bin,
                "ratio":  self.selection_ratio_bin,
                "mae":    self.mae_bin,
                "seng":   self.selected_engagement_bin,   # selected-set context
                "slen":   self.selected_length_bin,       # selected-set context
            },
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# SECTION 3 — MDP Environment
# ---------------------------------------------------------------------------

class PostSelectionEnvironment:
    """
    Episodic MDP environment for sequential post selection.

    One *episode* corresponds to processing all candidate posts for a
    *single user*.  The agent steps through the posts one at a time,
    deciding to select or skip each one.

    Reward (Sutton & Barto §3.2 — goals and rewards)
    -------------------------------------------------
    The reward signal is the *improvement* in downstream personality
    prediction quality caused by the most recent select action:

        R_{t+1} = (MAE_before − MAE_after) − selection_cost

    •  MAE_before − MAE_after > 0  →  the selected post *helped* prediction
    •  MAE_before − MAE_after < 0  →  the selected post *hurt* prediction
    •  selection_cost > 0          →  small per-selection penalty to discourage
                                       selecting everything indiscriminately
    •  skip action                 →  reward = 0 (no cost, no benefit)

    The downstream predictor is called via *predictor_fn*, an injected
    callable:
        predictor_fn(selected_posts: List[Dict]) -> float   (returns MAE)

    If predictor_fn is None (unit tests / development), a placeholder
    based on the average engagement of selected posts is used instead.

    Episode termination (§3.3 — returns and episodes)
    --------------------------------------------------
    Episode ends when:
        1. All candidate posts have been considered, OR
        2. n_selected reaches max_selected, OR
        3. Caller sets done explicitly via environment state.
    """

    ACTIONS = ("select", "skip")

    def __init__(
        self,
        posts: List[Dict[str, Any]],
        predictor_fn=None,
        selection_cost: float = 0.05,
        max_selected: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        posts          : Ordered list of candidate post feature dicts
                         (output of create_post_features).
        predictor_fn   : Callable[[List[Dict]], float] — returns MAE
                         when given the currently-selected post list.
                         Signature: predictor_fn(selected_posts) -> float

                         PRODUCTION: wire this to your frozen BERT → Lasso
                         pipeline evaluated on a held-out validation split:

                             def predictor_fn(selected_posts):
                                 embeddings = bert_encode(selected_posts)   # cached
                                 preds      = lasso.predict(embeddings)
                                 return mean_absolute_error(y_val, preds)

                         DEVELOPMENT: pass None to use the log-curve placeholder
                         (only safe for unit-testing; the agent learns nothing
                         meaningful about personality in that mode).

        selection_cost : Per-selection penalty subtracted from reward.
                         Prevents the degenerate "always select" policy.
                         Recommended range: 0.01 – 0.10.
        max_selected   : Hard limit on selections per episode.
        """
        self._posts = list(posts)
        self._predictor_fn = predictor_fn
        self._selection_cost = selection_cost
        self._max_selected = max_selected

        # Episode state — reset() initialises these properly
        self._cursor:         int = 0
        self._selected:       List[Dict] = []
        self._baseline_mae:   Optional[float] = None   # predictor_fn([])
        self._current_mae:    Optional[float] = None
        self._step_count:     int = 0
        self._done:           bool = False

    # ------------------------------------------------------------------
    # Public MDP interface (Sutton & Barto §3.6 finite MDPs)
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """
        Reset the environment for a new episode.

        FIX (Issue 1 — first-selection reward):
        We call predictor_fn([]) here — with an *empty* selected set — to
        establish a genuine baseline MAE *before* any posts are selected.

        Why this matters
        ----------------
        The old code set _current_mae = None and then gave the first selected
        post improvement = 0.0, meaning reward = -selection_cost regardless of
        how useful that post actually was.  The agent was explicitly taught
        "the first post you pick is bad" — a systematic training bias.

        With the fix:
            baseline_mae = predictor_fn([])   ← e.g. 1.8 (random/mean guess)
            After selecting post_1:
                mae_after   = predictor_fn([post_1])   ← e.g. 1.5
                improvement = 1.8 - 1.5 = +0.3
                reward      = 0.3 - selection_cost     ← correctly positive

        The predictor_fn([]) call is the personality model's error when it
        has *no posts at all* (pure prior / mean prediction).  Any post that
        beats that baseline earns positive reward from the very first step.

        Returns
        -------
        state_key : str — JSON-serialised initial state (Q-table key)
        """
        self._cursor = 0
        self._selected = []
        self._step_count = 0
        self._done = False

        # Establish baseline: MAE with zero posts selected.
        # This is the "no-information" error the agent must beat.
        self._baseline_mae = self._call_predictor_on([])
        self._current_mae = self._baseline_mae

        logger.debug(
            "Environment reset: %d candidate posts | baseline_mae=%.4f",
            len(self._posts),
            self._baseline_mae,
        )
        return self._observe()

    def step(self, action: str) -> Tuple[str, float, bool]:
        """
        Apply *action* to the current candidate post.

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
            raise RuntimeError("Episode is finished.  Call reset() first.")

        current_post = self._posts[self._cursor]
        mae_before = self._current_mae

        if action == "select":
            self._selected.append(current_post)
            mae_after = self._call_predictor_on(self._selected)
            reward = self._compute_reward(mae_before, mae_after)
            self._current_mae = mae_after
        else:
            # skip: zero reward — no cost, no benefit
            reward = 0.0

        self._cursor += 1
        self._step_count += 1

        self._done = (
            self._cursor >= len(self._posts)
            or len(self._selected) >= self._max_selected
        )

        next_state = self._observe()

        logger.debug(
            "Step %d | action=%s | reward=%.4f | done=%s | n_sel=%d | MAE=%.4f",
            self._step_count,
            action,
            reward,
            self._done,
            len(self._selected),
            self._current_mae if self._current_mae is not None else -1.0,
        )

        return next_state, reward, self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> str:
        """
        Build the current MDP state and return its Q-table key.

        FIX (Issue 4 — Markov state):
        Two additions make the state better approximate the Markov property:
            selected_engagement_bin — mean engagement tier of selected set
            selected_length_bin     — mean text-length tier of selected set

        Without these, two states with n_selected=3 look identical to the
        Q-table even if one selected set contains three viral long posts and
        the other contains three brief low-engagement posts.  The marginal
        value of the next post differs substantially in those two situations.
        Encoding the *composition* of the selected set, not just its count,
        gives the Q-table the information it needs to distinguish them.

        State-space size note
        ---------------------
        Adding 2 bins with 3–4 values each multiplies the theoretical state
        space by ~12.  In practice the Q-table only populates visited states,
        so the actual number of entries grows modestly with experience.
        This is the right trade-off: richer Markov approximation at tractable
        tabular cost.

        Terminal state
        --------------
        When the episode is finished (cursor past all posts or budget reached)
        we return a dedicated terminal-state key.  The Q-learning update for
        the last step uses this as next_state; its Q-values remain 0 by
        convention (no future reward after termination).
        """
        n_selected = len(self._selected)
        n_remaining = max(len(self._posts) - self._cursor, 0)
        total = n_selected + n_remaining
        ratio = n_selected / total if total > 0 else 0.0

        # Selected-set summary (for Markov approximation)
        sel_eng_bin = _bin_selected_engagement(self._selected)
        sel_len_bin = _bin_selected_length(self._selected)

        if self._done or self._cursor >= len(self._posts):
            # Terminal state: no current post.  Use neutral/zero bins for
            # current-post fields; keep selected-set summary accurate.
            return PostSelectionState(
                engagement_bin="low",
                recency_bin="old",
                length_bin="short",
                has_hashtags=False,
                has_urls=False,
                n_selected_bin=_bin_count(n_selected),
                n_remaining_bin="few",
                selection_ratio_bin=_bin_ratio(ratio),
                mae_bin=_bin_mae(self._current_mae),
                selected_engagement_bin=sel_eng_bin,
                selected_length_bin=sel_len_bin,
            ).to_key()

        post = self._posts[self._cursor]
        return PostSelectionState(
            engagement_bin=_bin_engagement(post.get("engagement_score", 0)),
            recency_bin=_bin_recency(post.get("recency_days", 0)),
            length_bin=_bin_length(post.get("text_length", 0)),
            has_hashtags=bool(post.get("has_hashtags", False)),
            has_urls=bool(post.get("has_urls", False)),
            n_selected_bin=_bin_count(n_selected),
            n_remaining_bin=_bin_count(n_remaining),
            selection_ratio_bin=_bin_ratio(ratio),
            mae_bin=_bin_mae(self._current_mae),
            selected_engagement_bin=sel_eng_bin,
            selected_length_bin=sel_len_bin,
        ).to_key()

    def _call_predictor_on(self, posts: List[Dict]) -> float:
        """
        Call the downstream personality predictor on an arbitrary *posts*
        list and return the resulting MAE.

        FIX (Issue 2 — real predictor connection):
        This method is now called with an explicit *posts* argument instead
        of always using self._selected.  This lets reset() call it with []
        to establish a proper baseline (Issue 1 fix) and lets step() call it
        with the updated selected set after each action.

        PRODUCTION WIRING
        -----------------
        Wire predictor_fn to your real pipeline like this:

            # Pre-compute BERT embeddings once per user (outside the RL loop)
            # so the predictor_fn only runs the cheap Lasso inference step.
            bert_cache = {post["id"]: bert_encode(post) for post in all_posts}

            def predictor_fn(selected_posts):
                if not selected_posts:
                    return baseline_mae   # mean predictor / prior
                X = np.vstack([bert_cache[p["id"]] for p in selected_posts])
                X_agg = X.mean(axis=0, keepdims=True)   # or concat+pool
                preds = lasso.predict(X_agg)             # shape (1, 5) for OCEAN
                return float(mean_absolute_error(y_val, preds))

        IMPORTANT — data-leakage note (Issue 3):
        The *y_val* used inside predictor_fn must come from a *held-out
        validation split* of users, not the same users whose posts are
        being stepped through.  See run_training_loop() docstring for the
        recommended train/val/test user split.

        DEVELOPMENT PLACEHOLDER
        -----------------------
        When predictor_fn is None the method uses a log-curve approximation:
            MAE ≈ 2.0 − 0.15 · log(1 + n)
        This is ONLY suitable for unit-testing the RL loop mechanics.
        It encodes "more posts → lower error" regardless of content, so
        the agent learns nothing about personality informativeness.
        Always supply a real predictor_fn before drawing conclusions.
        """
        if self._predictor_fn is not None:
            return float(self._predictor_fn(posts))

        # Development placeholder — do NOT use for real experiments
        n = len(posts)
        return max(0.1, 2.0 - 0.15 * math.log1p(n))

    def _compute_reward(self, mae_before: float, mae_after: float) -> float:
        """
        Compute reward for a 'select' action.

        FIX (Issue 1 — first-selection reward):
        mae_before is now always a real float (set to baseline_mae by reset()),
        never None.  The None branch that gave the first post zero improvement
        has been removed.

        Reward formula
        --------------
            R = (MAE_before − MAE_after) − selection_cost

        •  MAE_before − MAE_after > 0  →  post improved prediction  → positive R
        •  MAE_before − MAE_after < 0  →  post hurt prediction       → negative R
        •  MAE_before − MAE_after = 0  →  no change                  → −cost (negative)
        •  selection_cost              →  discourages selecting every post

        The baseline MAE established at reset() (predictor_fn([])) ensures the
        first selected post is measured against the *no-posts* error level, so
        a genuinely informative first post earns a large positive reward.
        """
        improvement = mae_before - mae_after   # positive ↔ better prediction
        return improvement - self._selection_cost

    # ------------------------------------------------------------------
    # Read-only accessors (useful for the training loop)
    # ------------------------------------------------------------------

    @property
    def selected_posts(self) -> List[Dict]:
        return list(self._selected)

    @property
    def current_mae(self) -> Optional[float]:
        return self._current_mae

    @property
    def is_done(self) -> bool:
        return self._done


# ---------------------------------------------------------------------------
# SECTION 4 — Tabular Q-Learning Agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning agent for active post selection.

    Implements the off-policy TD control algorithm from
    Sutton & Barto (2018), Chapter 6, Section 6.5:

        Q(S_t, A_t) ← Q(S_t, A_t)
                       + α [ R_{t+1} + γ · max_a Q(S_{t+1}, a) − Q(S_t, A_t) ]

    WHAT IS STORED IN THE Q-TABLE
    ------------------------------
    The Q-table maps (state_key, action) → expected cumulative reward.
    A high Q(s, 'select') means: "given the current episode context
    described by state s, selecting this post is expected to produce
    the most future improvement in Big Five prediction quality."

    EXPLORATION vs EXPLOITATION (§2.4, §6.5)
    ------------------------------------------
    ε-greedy: with probability ε the agent picks an action at random
    (exploration); otherwise it picks the action with the highest Q-value
    (exploitation).  ε is typically decayed over training.

    API COMPATIBILITY
    -----------------
    All original public methods (featurize_post, choose_action,
    update_q_value, select_posts, save_state, load_state) have identical
    signatures and return types to the original implementation so that
    existing Django pipeline code continues to work unchanged.
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
                  Small α → slow, stable learning.  Large α → fast but noisy.
        gamma   : Discount factor γ ∈ [0, 1).
                  How much the agent values future rewards vs immediate ones.
                  γ close to 1 → agent plans ahead (appropriate here because
                  later selections can compound earlier gains).
        epsilon : Exploration rate ε ∈ [0, 1].
                  Probability of choosing a random action rather than the
                  greedy best.  Set to 0 at evaluation time.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: state_key (str) → {action (str) → Q-value (float)}
        # Initialised to 0 (optimistic initialisation can be added later).
        self.q_table: Dict[str, Dict[str, float]] = {}
        # ['select', 'skip']
        self.actions = list(PostSelectionEnvironment.ACTIONS)

        logger.info(
            "QLearningAgent initialised | α=%.3f | γ=%.3f | ε=%.3f",
            alpha,
            gamma,
            epsilon,
        )

    # ------------------------------------------------------------------
    # State featurisation  (backward-compatible)
    # ------------------------------------------------------------------

    def featurize_post(self, post_data: Dict) -> str:
        """
        Convert a single post feature dict to a Q-table state key.

        BACKWARD COMPATIBILITY NOTE
        ---------------------------
        This method is retained for code that calls it directly (e.g.
        existing Django views that want a quick state hash for logging).

        It produces a *post-only* state with all episode-context fields
        set to their neutral/zero defaults:
            n_selected  = "few",  n_remaining = "some",  ratio = "low"
            mae         = "none", selected_engagement = "none"
            selected_length = "none"

        This means the returned key will NOT match any key the agent was
        trained on (which always had real episode context).  Therefore:

        • DO use for: logging, quick per-post diagnostics, legacy callers.
        • DO NOT use for: actual inference / post selection decisions.

        For real inference, always use select_posts(), which runs through
        PostSelectionEnvironment and produces consistent episode-context
        states (Issue 5 fix).

        Parameters
        ----------
        post_data : Dict with keys: engagement_score, recency_days,
                    text_length, has_hashtags, has_urls.

        Returns
        -------
        JSON string — Q-table key (same schema as PostSelectionState.to_key())
        """
        return PostSelectionState(
            engagement_bin=_bin_engagement(
                post_data.get("engagement_score", 0)),
            recency_bin=_bin_recency(post_data.get("recency_days", 0)),
            length_bin=_bin_length(post_data.get("text_length", 0)),
            has_hashtags=bool(post_data.get("has_hashtags", False)),
            has_urls=bool(post_data.get("has_urls", False)),
            n_selected_bin="few",
            n_remaining_bin="some",
            selection_ratio_bin="low",
            mae_bin="none",
            selected_engagement_bin="none",
            selected_length_bin="none",
        ).to_key()

    # ------------------------------------------------------------------
    # Q-table accessors
    # ------------------------------------------------------------------

    def get_q_value(self, state: str, action: str) -> float:
        """
        Return Q(state, action).  Unseen (s, a) pairs are initialised to 0.0.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state].get(action, 0.0)

    def _ensure_state(self, state: str) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    # ------------------------------------------------------------------
    # Action selection — ε-greedy (Sutton & Barto §2.4)
    # ------------------------------------------------------------------

    def choose_action(self, state: str, training: bool = True) -> str:
        """
        Select an action using ε-greedy policy.

        During *training*:
            With probability ε  → random action (explore new state-action pairs)
            With probability 1−ε → action with highest Q-value (exploit)

        During *evaluation* (training=False):
            Always greedy — pick action with highest Q-value.
            Ties broken randomly.

        Parameters
        ----------
        state    : Q-table key (output of featurize_post or env._observe())
        training : If False, acts purely greedily (ε=0).

        Returns
        -------
        'select' or 'skip'
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: uniformly random action
            chosen = np.random.choice(self.actions)
            logger.debug("ε-greedy EXPLORE → %s", chosen)
            return chosen

        # Exploitation: argmax_a Q(state, a)
        self._ensure_state(state)
        q_vals = self.q_table[state]
        max_q = max(q_vals.values())
        # Break ties randomly for unbiased greedy selection
        best = [a for a, v in q_vals.items() if v == max_q]
        chosen = np.random.choice(best)
        logger.debug("ε-greedy EXPLOIT → %s (Q=%.4f)", chosen, max_q)
        return chosen

    # ------------------------------------------------------------------
    # Q-Learning update  (Sutton & Barto §6.5, eq. 6.8)
    # ------------------------------------------------------------------

    def update_q_value(
        self,
        state:      str,
        action:     str,
        reward:     float,
        next_state: str,
    ) -> Tuple[float, float]:
        """
        Apply the tabular Q-learning update rule:

            Q(S_t, A_t) ← Q(S_t, A_t)
                           + α [R_{t+1} + γ · max_a Q(S_{t+1}, a) − Q(S_t, A_t)]

        This is an *off-policy* TD(0) update: the max over next-state
        actions makes the target the greedy (optimal) estimate regardless
        of which action was actually taken next.  This is what separates
        Q-learning from Sarsa (on-policy).

        Parameters
        ----------
        state      : S_t  — current state key
        action     : A_t  — action taken
        reward     : R_{t+1} — reward received
        next_state : S_{t+1} — successor state key

        Returns
        -------
        (old_q_value, new_q_value) — for logging and monitoring.
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())  # max_a Q(S', a)

        # Bellman update (off-policy TD target)
        td_error = reward + self.gamma * max_next_q - old_q
        new_q = old_q + self.alpha * td_error

        self.q_table[state][action] = new_q

        logger.debug(
            "Q-update | s=%s… | a=%s | R=%.4f | Q: %.4f → %.4f | δ=%.4f",
            state[:40],
            action,
            reward,
            old_q,
            new_q,
            td_error,
        )

        return old_q, new_q

    # ------------------------------------------------------------------
    # Batch inference — backward-compatible with original pipeline
    # ------------------------------------------------------------------

    def select_posts(
        self,
        posts: List[Dict],
        top_k: int = 10,
        training: bool = False,
        predictor_fn=None,
        selection_cost: float = 0.05,
    ) -> List[Dict]:
        """
        Select up to *top_k* posts using the trained Q-policy.

        FIX (Issue 5 — train/inference state consistency):
        The previous implementation called featurize_post() on each post in
        isolation, producing states with hardcoded n_selected='few',
        n_remaining='some', ratio='low', mae='none' — a completely different
        state representation from what the agent was trained on.

        Training used:  Q(s_full_context, action)
        Inference used: Q(s_post_only,    action)   ← WRONG: different state

        The fix: inference now steps through the *same* PostSelectionEnvironment
        used during training.  Every state the agent observes at inference time
        is built by exactly the same _observe() logic, so the Q-table is queried
        with keys it actually saw during training.

        No Q-learning updates are performed (training=False → greedy policy,
        no Bellman update called).  The environment still needs to produce
        states (it does not need to compute rewards unless you want them).

        HOW THE TOP-K LIMIT WORKS
        --------------------------
        top_k is enforced via the environment's max_selected parameter.  The
        agent may select fewer than top_k posts if the greedy policy judges
        remaining candidates unworthy (Q(s, 'skip') > Q(s, 'select')).

        Parameters
        ----------
        posts          : List of post feature dicts (output of create_post_features)
        top_k          : Maximum number of posts to return
        training       : If True, uses ε-greedy; if False (default), acts greedily
        predictor_fn   : Optional downstream predictor for reward-aware state
                         (mae_bin in state will be more accurate if supplied).
                         If None, baseline_mae defaults to the log placeholder.
        selection_cost : Forwarded to the environment (must match training value).

        Returns
        -------
        List of selected post dicts, each augmented with:
            q_value : float — Q(state, 'select') at the moment of selection
            state   : str   — full Q-table key (including episode context)
            action  : str   — always 'select'
        Ordered by selection sequence (the order the agent chose them),
        not by Q-value, because the episode context changes after each
        selection and Q-values are not directly comparable across steps.
        """
        env = PostSelectionEnvironment(
            posts=posts,
            predictor_fn=predictor_fn,
            selection_cost=selection_cost,
            max_selected=top_k,
        )

        state = env.reset()
        selected: List[Dict] = []

        while not env.is_done:
            action = self.choose_action(state, training=training)
            q_val = self.get_q_value(state, "select")

            # Step the environment (advances cursor, updates selected set & state)
            next_state, _reward, done = env.step(action)

            if action == "select":
                # The post that was just processed is env._selected[-1]
                chosen_post = env.selected_posts[-1]
                selected.append(
                    {
                        **chosen_post,
                        "q_value": q_val,
                        "state":   state,
                        "action":  "select",
                    }
                )

            state = next_state

        logger.info(
            "select_posts | selected %d / %d | top_k=%d",
            len(selected),
            len(posts),
            top_k,
        )
        return selected

    # ------------------------------------------------------------------
    # Persistence — backward-compatible with original Django serialisers
    # ------------------------------------------------------------------

    def save_state(self) -> Dict:
        """
        Serialise the full agent state to a plain dict (JSON-safe).

        Compatible with the original save_state() schema so existing
        Django model fields / cache storage code works unchanged.
        """
        return {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "alpha":   self.alpha,
            "gamma":   self.gamma,
            "epsilon": self.epsilon,
        }

    def load_state(self, state_dict: Dict) -> None:
        """
        Restore agent state from a previously serialised dict.

        Compatible with original load_state() schema.
        """
        self.q_table = {k: dict(v)
                        for k, v in state_dict.get("q_table", {}).items()}
        self.alpha = state_dict.get("alpha",   self.alpha)
        self.gamma = state_dict.get("gamma",   self.gamma)
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
    episode:        int
    total_reward:   float
    n_selected:     int
    n_candidates:   int
    final_mae:      Optional[float]
    n_q_updates:    int
    mean_td_error:  float


def run_training_episode(
    agent:   QLearningAgent,
    env:     PostSelectionEnvironment,
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
    agent   : QLearningAgent — will be updated in-place
    env     : PostSelectionEnvironment — will be reset at the start
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
        # --- Action selection (ε-greedy) ---
        action = agent.choose_action(state, training=True)

        # --- Environment transition ---
        next_state, reward, done = env.step(action)

        # --- Q-learning update (only when meaningful) ---
        # We update for both select and skip so the agent learns the value
        # of restraint (skip → reward=0) as well as selection.
        old_q, new_q = agent.update_q_value(state, action, reward, next_state)
        td_errors.append(abs(new_q - old_q))

        total_reward += reward
        state = next_state

    stats = EpisodeStats(
        episode=episode,
        total_reward=total_reward,
        n_selected=len(env.selected_posts),
        n_candidates=len(env._posts),
        final_mae=env.current_mae,
        n_q_updates=len(td_errors),
        mean_td_error=float(np.mean(td_errors)) if td_errors else 0.0,
    )

    logger.info(
        "Episode %4d | R=%.4f | selected=%d/%d | MAE=%.4f | "
        "Q-updates=%d | mean|δ|=%.4f | Q-table=%d states",
        episode,
        stats.total_reward,
        stats.n_selected,
        stats.n_candidates,
        stats.final_mae or -1.0,
        stats.n_q_updates,
        stats.mean_td_error,
        len(agent.q_table),
    )

    return stats


def run_training_loop(
    agent:              QLearningAgent,
    user_post_batches:  List[List[Dict]],
    predictor_fn=None,
    n_epochs:           int = 5,
    selection_cost:     float = 0.05,
    max_selected:       int = 20,
    epsilon_start:      float = 1.0,
    epsilon_end:        float = 0.05,
    epsilon_decay:      float = 0.995,
) -> List[EpisodeStats]:
    """
    Full Q-learning training loop (Sutton & Barto §6.5).

    WHAT THIS LOOP LEARNS
    ---------------------
    The agent iterates over every user's post collection (*user_post_batches*)
    for *n_epochs* epochs.  In each episode it tries different combinations
    of posts via ε-greedy exploration.  The reward signal from the downstream
    predictor (*predictor_fn*) teaches the agent which post features (or
    combinations thereof) reliably reduce personality-prediction MAE.

    After training, agent.select_posts() embodies the learned policy: it
    picks posts whose state representation has high Q(s, 'select') — i.e.
    posts that the agent has learned to associate with improvements in
    personality prediction.

    ε-DECAY SCHEDULE
    ----------------
    epsilon is decayed multiplicatively after every episode:
        ε ← max(ε_end, ε · decay)
    This realises the standard exploration → exploitation annealing
    described in Sutton & Barto §2.5.

    Parameters
    ----------
    agent              : QLearningAgent — updated in-place
    user_post_batches  : List of per-user post lists (each is one episode)
    predictor_fn       : Callable[[List[Dict]], float] → MAE, or None
    n_epochs           : How many full passes over all users
    selection_cost     : Per-selection penalty forwarded to the environment
    max_selected       : Maximum selections per episode
    epsilon_start      : Initial ε (usually 1.0 = full exploration)
    epsilon_end        : Minimum ε (usually ~0.05 = mostly greedy)
    epsilon_decay      : Multiplicative decay applied after each episode

    Returns
    -------
    List[EpisodeStats] — one entry per episode (epoch × n_users)
    """
    agent.epsilon = epsilon_start
    all_stats: List[EpisodeStats] = []
    episode_idx = 0

    for epoch in range(n_epochs):
        epoch_reward = 0.0

        for user_posts in user_post_batches:
            env = PostSelectionEnvironment(
                posts=user_posts,
                predictor_fn=predictor_fn,
                selection_cost=selection_cost,
                max_selected=max_selected,
            )

            stats = run_training_episode(agent, env, episode=episode_idx)
            all_stats.append(stats)
            epoch_reward += stats.total_reward
            episode_idx += 1

            # ε-decay: less exploration as training progresses
            agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        logger.info(
            "Epoch %d/%d complete | mean episode reward=%.4f | ε=%.4f | Q-table=%d",
            epoch + 1,
            n_epochs,
            epoch_reward / max(len(user_post_batches), 1),
            agent.epsilon,
            len(agent.q_table),
        )

    logger.info(
        "Training complete | total episodes=%d | final Q-table=%d states",
        episode_idx,
        len(agent.q_table),
    )
    return all_stats


# ---------------------------------------------------------------------------
# SECTION 6 — Django model integration helper (backward-compatible)
# ---------------------------------------------------------------------------

def create_post_features(post_obj) -> Dict:
    """
    Convert a Django Post model instance to a feature dict for the agent.

    This function is the bridge between the Django ORM and the RL pipeline.
    It produces the dict format consumed by both featurize_post() and the
    PostSelectionEnvironment.

    Parameters
    ----------
    post_obj : Django Post model instance with fields:
        created_at_original : datetime (timezone-aware)
        engagement_score    : float
        content             : str (raw)
        cleaned_content     : str (optional; falls back to content)

    Returns
    -------
    Dict with keys:
        engagement_score  : float
        recency_days      : int
        text_length       : int  (characters in cleaned content)
        has_hashtags      : bool
        has_urls          : bool
    """
    from datetime import datetime, timezone

    recency_days = (datetime.now(timezone.utc) -
                    post_obj.created_at_original).days
    content_text = getattr(post_obj, "cleaned_content", post_obj.content)

    return {
        "engagement_score": post_obj.engagement_score,
        "recency_days":     recency_days,
        "text_length":      len(content_text),
        "has_hashtags":     "#" in post_obj.content,
        "has_urls":         "http" in post_obj.content.lower(),
    }
