"""
Deterministic interpretation layer for PANDORA experiment outputs.

Consumes structured results from ``metrics_engine.evaluate()`` and
``experiments.pandora_runner`` (bundle keys: results, findings, factor_effects,
data_quality, hybrid_cell_evaluations, config, sample, reproducibility).

This module does not:
- recalculate MAE, R², F1, ROC, PR, or confidence scores
- call an LLM or any external API
- issue clinical or psychological diagnoses

All labels come from explicit, configurable reporting bands. Those bands are
project conventions for writing up a run, not published psychometric cutoffs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

TRAIT_NAMES: Dict[str, str] = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
TRAIT_ORDER: Tuple[str, ...] = ("O", "C", "E", "A", "N")
QUALITY_RANK = {"poor": 0, "limited": 1, "moderate": 2, "strong": 3}
EFFECT_LABELS = ("improved", "reduced", "had_minimal_effect", "produced_a_mixed_effect")


@dataclass
class InterpretationThresholds:
    """
    Explicit reporting bands. Override any field to change the language
    without changing metric calculations.

    Prediction bands apply to normalized [0, 1] OCEAN model outputs.
    Quality bands apply to already-computed metrics (MAE lower-is-better;
    R² / F1 / ROC-AUC higher-is-better).
    Evidence bands apply to participant and comment counts already recorded
    by the experiment / data-quality layers.
    """

    # A. Normalized prediction position. Each tuple is (upper_inclusive, code, phrase).
    prediction_bands: Tuple[Tuple[float, str, str], ...] = (
        (0.20, "lower_end", "lower end"),
        (0.40, "below_midpoint", "below the midpoint"),
        (0.60, "around_midpoint", "around the midpoint"),
        (0.80, "higher_end", "higher end"),
        (1.00, "upper_end", "upper end"),
    )

    # B. Continuous quality (MAE on the unit scale used by the runner).
    mae_strong_max: float = 0.10
    mae_moderate_max: float = 0.20
    mae_limited_max: float = 0.30

    r2_strong_min: float = 0.50
    r2_moderate_min: float = 0.25
    r2_limited_min: float = 0.10

    f1_strong_min: float = 0.80
    f1_moderate_min: float = 0.65
    f1_limited_min: float = 0.50

    roc_auc_strong_min: float = 0.80
    roc_auc_moderate_min: float = 0.70
    roc_auc_limited_min: float = 0.60

    # C. Evidence (counts already measured; not inferred from text).
    participants_insufficient_max: int = 10
    participants_limited_max: int = 30
    participants_moderate_max: int = 80
    mean_comments_insufficient_max: float = 3.0
    mean_comments_limited_max: float = 6.0
    mean_comments_moderate_max: float = 12.0

    # D. Effect size below which a matched-pair delta is called minimal.
    mae_minimal_abs: float = 0.01
    r2_minimal_abs: float = 0.02

    # Small-sample qualification (held-out test participants).
    small_test_n: int = 20

    notes: Tuple[str, ...] = (
        "These bands are deterministic reporting conventions for this project.",
        "They are not clinical cutoffs and not claims from an external validation study.",
    )


DEFAULT_THRESHOLDS = InterpretationThresholds()


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str, dict, list)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _f(value)
    return value


def _as_records(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    rows: List[Any] = []
    if hasattr(obj, "to_dict"):
        try:
            rows = list(obj.to_dict("records"))
        except Exception:
            rows = []
    elif isinstance(obj, list):
        rows = [row for row in obj if isinstance(row, Mapping)]
    elif isinstance(obj, Mapping) and {"delta_mae", "model"} <= set(obj.keys()):
        rows = [dict(obj)]
    return [
        {str(k): _plain(v) for k, v in row.items()}
        for row in rows
        if isinstance(row, Mapping)
    ]


def _f(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _worse(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if QUALITY_RANK.get(a, 0) <= QUALITY_RANK.get(b, 0) else b


def _trait_name(key: str) -> str:
    return TRAIT_NAMES.get(key, key)


def _label_higher_is_better(value: Optional[float], strong: float, moderate: float, limited: float) -> Optional[str]:
    if value is None:
        return None
    if value >= strong:
        return "strong"
    if value >= moderate:
        return "moderate"
    if value >= limited:
        return "limited"
    return "poor"


def _label_mae(value: Optional[float], t: InterpretationThresholds) -> Optional[str]:
    if value is None:
        return None
    if value <= t.mae_strong_max:
        return "strong"
    if value <= t.mae_moderate_max:
        return "moderate"
    if value <= t.mae_limited_max:
        return "limited"
    return "poor"


def _label_participants(n: Optional[int], t: InterpretationThresholds) -> Optional[str]:
    if n is None:
        return None
    if n < t.participants_insufficient_max:
        return "insufficient_evidence"
    if n < t.participants_limited_max:
        return "limited_evidence"
    if n < t.participants_moderate_max:
        return "moderate_evidence"
    return "strong_evidence"


def _label_mean_comments(mean_n: Optional[float], t: InterpretationThresholds) -> Optional[str]:
    if mean_n is None:
        return None
    if mean_n < t.mean_comments_insufficient_max:
        return "insufficient_evidence"
    if mean_n < t.mean_comments_limited_max:
        return "limited_evidence"
    if mean_n < t.mean_comments_moderate_max:
        return "moderate_evidence"
    return "strong_evidence"


def _evidence_rank(label: Optional[str]) -> int:
    order = {
        "insufficient_evidence": 0,
        "limited_evidence": 1,
        "moderate_evidence": 2,
        "strong_evidence": 3,
    }
    return order.get(label or "", -1)


def _worse_evidence(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if _evidence_rank(a) <= _evidence_rank(b) else b


# ---------------------------------------------------------------------------
# A. Prediction interpretation
# ---------------------------------------------------------------------------

def interpret_prediction(
    value: float,
    trait_key: str,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Describe one normalized model output as a position on the [0, 1] scale.

    This is a model prediction description, not a personality diagnosis.
    """
    score = float(value)
    code, phrase = "out_of_range", "outside the normalized scale"
    for upper, band_code, band_phrase in thresholds.prediction_bands:
        if score <= upper + 1e-12:
            code, phrase = band_code, band_phrase
            break
    name = _trait_name(trait_key)
    statement = (
        f"The model predicted a value toward the {phrase} of the normalized "
        f"{name} scale ({score:.2f}). This is a model output, not a diagnosis."
    )
    return {
        "trait_key": trait_key,
        "trait_name": name,
        "predicted_value": round(score, 4),
        "band": code,
        "phrase": phrase,
        "statement": statement,
        "is_diagnosis": False,
        "is_clinical_claim": False,
    }


def interpret_ocean_predictions(
    predictions: Union[Mapping[str, Any], Sequence[Any]],
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Interpret a 5-trait model output. Accepts O/C/E/A/N keys or a length-5 sequence."""
    values: Dict[str, Optional[float]] = {}
    if isinstance(predictions, Mapping):
        for key in TRAIT_ORDER:
            values[key] = _f(predictions.get(key))
            if values[key] is None:
                values[key] = _f(predictions.get(TRAIT_NAMES[key]))
    else:
        seq = list(predictions)
        if len(seq) != 5:
            raise ValueError("interpret_ocean_predictions expects 5 values or an O/C/E/A/N mapping.")
        for key, raw in zip(TRAIT_ORDER, seq):
            values[key] = _f(raw)

    per_trait = {}
    for key in TRAIT_ORDER:
        if values[key] is None:
            continue
        per_trait[key] = interpret_prediction(values[key], key, thresholds)

    return {
        "kind": "prediction_interpretation",
        "per_trait": per_trait,
        "statements": [per_trait[k]["statement"] for k in TRAIT_ORDER if k in per_trait],
        "disclaimer": (
            "These statements describe where the model's normalized scores sit "
            "on [0, 1]. They are not assessments of a person's personality."
        ),
    }


# ---------------------------------------------------------------------------
# B. Prediction-quality interpretation
# ---------------------------------------------------------------------------

def interpret_prediction_quality(
    regression: Optional[Mapping[str, Any]] = None,
    binary: Optional[Mapping[str, Any]] = None,
    *,
    mae: Optional[float] = None,
    r2: Optional[float] = None,
    f1: Optional[float] = None,
    roc_auc: Optional[float] = None,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
    source: str = "supplied_metrics",
) -> Dict[str, Any]:
    """
    Label already-computed metrics. Does not calculate MAE/R²/F1/AUC.

    ``regression`` / ``binary`` may be a metrics_engine block
    (``{per_trait, aggregate}``) or a runner ``overall`` dict.
    """
    if regression:
        agg = regression.get("aggregate") if "aggregate" in regression else regression
        mae = mae if mae is not None else _f((agg or {}).get("mae"))
        r2 = r2 if r2 is not None else _f((agg or {}).get("r2"))
    if binary:
        bagg = binary.get("aggregate") if "aggregate" in binary else binary
        f1 = f1 if f1 is not None else _f((bagg or {}).get("f1") or (bagg or {}).get("official_f1"))
        roc_auc = roc_auc if roc_auc is not None else _f((bagg or {}).get("roc_auc"))

    mae_label = _label_mae(mae, thresholds)
    r2_label = _label_higher_is_better(
        r2, thresholds.r2_strong_min, thresholds.r2_moderate_min, thresholds.r2_limited_min,
    )
    f1_label = _label_higher_is_better(
        f1, thresholds.f1_strong_min, thresholds.f1_moderate_min, thresholds.f1_limited_min,
    )
    roc_label = _label_higher_is_better(
        roc_auc, thresholds.roc_auc_strong_min, thresholds.roc_auc_moderate_min, thresholds.roc_auc_limited_min,
    )
    continuous_label = _worse(mae_label, r2_label)

    per_trait: Dict[str, Any] = {}
    if regression and isinstance(regression.get("per_trait"), Mapping):
        for key, block in regression["per_trait"].items():
            if not isinstance(block, Mapping):
                continue
            t_mae = _label_mae(_f(block.get("mae")), thresholds)
            t_r2 = _label_higher_is_better(
                _f(block.get("r2")),
                thresholds.r2_strong_min, thresholds.r2_moderate_min, thresholds.r2_limited_min,
            )
            per_trait[key] = {
                "mae": _f(block.get("mae")),
                "r2": _f(block.get("r2")),
                "mae_label": t_mae,
                "r2_label": t_r2,
                "continuous_label": _worse(t_mae, t_r2),
            }

    return {
        "kind": "prediction_quality_interpretation",
        "source": source,
        "metrics_read": {"mae": mae, "r2": r2, "f1": f1, "roc_auc": roc_auc},
        "labels": {
            "mae": mae_label,
            "r2": r2_label,
            "f1": f1_label,
            "roc_auc": roc_label,
            "continuous": continuous_label,
        },
        "continuous_rule": (
            "MAE and R² are labeled separately, then the more conservative "
            "(worse) of the two is reported as continuous."
        ),
        "per_trait": per_trait,
        "statement": _quality_statement(continuous_label, mae, r2, f1, roc_auc),
        "thresholds": {
            "mae_strong_max": thresholds.mae_strong_max,
            "mae_moderate_max": thresholds.mae_moderate_max,
            "mae_limited_max": thresholds.mae_limited_max,
            "r2_strong_min": thresholds.r2_strong_min,
            "r2_moderate_min": thresholds.r2_moderate_min,
            "r2_limited_min": thresholds.r2_limited_min,
            "f1_strong_min": thresholds.f1_strong_min,
            "f1_moderate_min": thresholds.f1_moderate_min,
            "f1_limited_min": thresholds.f1_limited_min,
        },
    }


def _quality_statement(
    continuous_label: Optional[str],
    mae: Optional[float],
    r2: Optional[float],
    f1: Optional[float],
    roc_auc: Optional[float],
) -> str:
    bits = []
    if continuous_label:
        bits.append(
            f"On the project's reporting bands, continuous prediction quality is {continuous_label}"
        )
        extras = []
        if mae is not None:
            extras.append(f"MAE={mae:.3f}")
        if r2 is not None:
            extras.append(f"R²={r2:.3f}")
        if extras:
            bits[-1] += " (" + ", ".join(extras) + ")"
        bits[-1] += "."
    if f1 is not None:
        bits.append(f"Official binary F1 read from the metrics was {f1:.3f}.")
    if roc_auc is not None:
        bits.append(f"ROC-AUC read from the metrics was {roc_auc:.3f}.")
    bits.append(
        "These labels follow documented project bands and are not an external "
        "scientific claim of generalisation."
    )
    return " ".join(bits) if bits else "No regression or binary metrics were available to interpret."


# ---------------------------------------------------------------------------
# C. Evidence-quality interpretation
# ---------------------------------------------------------------------------

def interpret_evidence_quality(
    *,
    n_participants: Optional[int] = None,
    n_test: Optional[int] = None,
    n_train: Optional[int] = None,
    n_comments: Optional[int] = None,
    mean_comments: Optional[float] = None,
    data_quality: Optional[Mapping[str, Any]] = None,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
    population: str = "specified_counts",
) -> Dict[str, Any]:
    """
    Label usable evidence from already-recorded counts.

    Prefer ``data_quality`` from the runner when present. Does not recount rows.
    """
    dq = data_quality or {}
    filt = dq.get("experiment_filter") or {}
    volume = dq.get("comment_volume") or {}
    train_vol = volume.get("train") or {}
    test_vol = volume.get("test") or {}
    used_vol = volume.get("sampled_all_folds") or {}

    if n_participants is None:
        n_participants = _i(filt.get("users_used"))
    if n_train is None:
        n_train = _i(train_vol.get("n_participants"))
    if n_test is None:
        n_test = _i(test_vol.get("n_participants"))
    if mean_comments is None:
        mean_comments = _f(train_vol.get("mean_comments_per_participant"))
        if mean_comments is None:
            mean_comments = _f(used_vol.get("mean_comments_per_participant"))
    if n_comments is None:
        n_comments = _i(train_vol.get("n_comments"))

    participant_n = n_train if n_train is not None else n_participants
    participant_label = _label_participants(participant_n, thresholds)
    test_label = _label_participants(n_test, thresholds)
    comment_label = _label_mean_comments(mean_comments, thresholds)
    overall = _worse_evidence(participant_label, comment_label)

    small_test = n_test is not None and n_test < thresholds.small_test_n

    statement = (
        f"Recorded usable evidence is labeled {overall or 'unavailable'} "
        f"under the project's count bands"
    )
    details = []
    if participant_n is not None:
        details.append(f"{participant_n} training/used participants ({participant_label})")
    if n_test is not None:
        details.append(f"{n_test} held-out test participants ({test_label})")
    if mean_comments is not None:
        details.append(f"mean {mean_comments:.2f} comments/participant ({comment_label})")
    if details:
        statement += " (" + "; ".join(details) + ")."
    else:
        statement += "; participant/comment counts were not supplied."
    if small_test:
        statement += (
            f" The held-out set is smaller than the configured small-sample "
            f"threshold ({thresholds.small_test_n}); findings from this run "
            f"should be treated as descriptive of this sample only."
        )

    return {
        "kind": "evidence_quality_interpretation",
        "population": population,
        "counts": {
            "n_participants": n_participants,
            "n_train": n_train,
            "n_test": n_test,
            "n_comments": n_comments,
            "mean_comments_per_participant": mean_comments,
        },
        "labels": {
            "participants": participant_label,
            "test_participants": test_label,
            "mean_comments": comment_label,
            "overall": overall,
        },
        "small_test_set": small_test,
        "small_test_threshold": thresholds.small_test_n,
        "statement": statement,
        "thresholds": {
            "participants_insufficient_max": thresholds.participants_insufficient_max,
            "participants_limited_max": thresholds.participants_limited_max,
            "participants_moderate_max": thresholds.participants_moderate_max,
            "mean_comments_insufficient_max": thresholds.mean_comments_insufficient_max,
            "mean_comments_limited_max": thresholds.mean_comments_limited_max,
            "mean_comments_moderate_max": thresholds.mean_comments_moderate_max,
        },
    }


# ---------------------------------------------------------------------------
# D. Experiment-effect interpretation
# ---------------------------------------------------------------------------

def _effect_from_delta_mae(
    delta_mae: Optional[float],
    delta_r2: Optional[float],
    thresholds: InterpretationThresholds,
) -> str:
    """
    MAE: negative delta means the treatment had lower error than the control
    (runner stores treatment minus baseline). R²: positive delta is an improvement.
    """
    mae = _f(delta_mae)
    r2 = _f(delta_r2)
    mae_dir = None
    if mae is not None:
        if abs(mae) < thresholds.mae_minimal_abs:
            mae_dir = "had_minimal_effect"
        elif mae < 0:
            mae_dir = "improved"
        else:
            mae_dir = "reduced"
    r2_dir = None
    if r2 is not None:
        if abs(r2) < thresholds.r2_minimal_abs:
            r2_dir = "had_minimal_effect"
        elif r2 > 0:
            r2_dir = "improved"
        else:
            r2_dir = "reduced"
    if mae_dir and r2_dir and mae_dir != r2_dir and "had_minimal_effect" not in (mae_dir, r2_dir):
        return "produced_a_mixed_effect"
    if mae_dir and r2_dir and mae_dir != r2_dir:
        return "produced_a_mixed_effect"
    return mae_dir or r2_dir or "had_minimal_effect"


def _summarize_cell_effects(
    rows: Sequence[Mapping[str, Any]],
    thresholds: InterpretationThresholds,
) -> str:
    labels = [
        _effect_from_delta_mae(row.get("delta_mae"), row.get("delta_r2"), thresholds)
        for row in rows
    ]
    if not labels:
        return "had_minimal_effect"
    improved = any(x == "improved" for x in labels)
    reduced = any(x == "reduced" for x in labels)
    mixed_cell = any(x == "produced_a_mixed_effect" for x in labels)
    if mixed_cell or (improved and reduced):
        return "produced_a_mixed_effect"
    if improved:
        return "improved"
    if reduced:
        return "reduced"
    return "had_minimal_effect"


def _interaction_block(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    group_key: str,
    thresholds: InterpretationThresholds,
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = row.get(group_key)
        groups.setdefault("none" if key is None else str(key), []).append(row)
    by_level = {
        level: _summarize_cell_effects(level_rows, thresholds)
        for level, level_rows in groups.items()
    }
    unique = set(by_level.values())
    depends = len(unique) > 1
    effect = next(iter(unique)) if len(unique) == 1 else "produced_a_mixed_effect"
    levels = ", ".join(f"{level}={label}" for level, label in by_level.items())
    if depends:
        statement = (
            f"{name} differed by {group_key} ({levels}). "
            "This is a descriptive observation of recorded deltas, not a statistical interaction test."
        )
    else:
        statement = (
            f"{name} was consistent across {group_key} ({levels or 'no matched pairs'}). "
            "This is a descriptive observation of recorded deltas, not a statistical interaction test."
        )
    return {
        "name": name,
        "grouped_by": group_key,
        "effect": effect,
        "depends_on_other_factor": depends,
        "by_level": by_level,
        "statement": statement,
    }


def _model_pair_rows(model_comparison: Any) -> List[Dict[str, Any]]:
    rows = []
    for row in _as_records(model_comparison):
        lasso_mae = _f(row.get("lasso_mae"))
        lstm_mae = _f(row.get("lstm_mae"))
        lasso_r2 = _f(row.get("lasso_r2"))
        lstm_r2 = _f(row.get("lstm_r2"))
        rows.append({
            "selection": row.get("selection"),
            "gan": row.get("gan"),
            "lasso_mae": lasso_mae,
            "lstm_mae": lstm_mae,
            "lasso_r2": lasso_r2,
            "lstm_r2": lstm_r2,
            "delta_mae": None if lasso_mae is None or lstm_mae is None else lstm_mae - lasso_mae,
            "delta_r2": None if lasso_r2 is None or lstm_r2 is None else lstm_r2 - lasso_r2,
        })
    return rows


def interpret_experiment_effects(
    bundle: Optional[Mapping[str, Any]] = None,
    *,
    results: Optional[Mapping[str, Any]] = None,
    findings: Optional[Mapping[str, Any]] = None,
    factor_effects: Optional[Mapping[str, Any]] = None,
    n_test: Optional[int] = None,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Interpret precomputed factorial comparisons. Does not recompute deltas.

    Reads runner ``factor_effects`` / ``findings`` / per-condition ``overall``.
    """
    bundle = bundle or {}
    results = results or bundle.get("results") or {}
    findings = findings or bundle.get("findings") or {}
    effects = factor_effects or bundle.get("factor_effects") or {}
    sample = bundle.get("sample") or {}
    if n_test is None:
        n_test = _i(sample.get("n_test"))

    q_rows = _as_records(effects.get("qlearning_effect"))
    g_rows = _as_records(effects.get("gan_effect"))
    if not q_rows and findings.get("qlearning_effect_mean"):
        q_rows = [dict(findings["qlearning_effect_mean"], model="mean", gan=None)]
    if not g_rows and findings.get("gan_effect_mean"):
        g_rows = [dict(findings["gan_effect_mean"], model="mean", selection=None)]

    q_label = _summarize_cell_effects(q_rows, thresholds)
    g_label = _summarize_cell_effects(g_rows, thresholds)

    model_means = findings.get("model_means") or {}
    lasso_mae = _f((model_means.get("lasso") or {}).get("mae"))
    lstm_mae = _f((model_means.get("lstm") or {}).get("mae"))
    lasso_r2 = _f((model_means.get("lasso") or {}).get("r2"))
    lstm_r2 = _f((model_means.get("lstm") or {}).get("r2"))
    mae_delta = None if lasso_mae is None or lstm_mae is None else lstm_mae - lasso_mae
    r2_delta = None if lasso_r2 is None or lstm_r2 is None else lstm_r2 - lasso_r2
    # Treat Lasso as the reference: negative mae_delta => LSTM improved (lower MAE).
    model_pair_rows = _model_pair_rows(bundle.get("model_comparison"))
    if model_pair_rows:
        model_label = _summarize_cell_effects(model_pair_rows, thresholds)
    else:
        model_label = _effect_from_delta_mae(mae_delta, r2_delta, thresholds)

    interactions = [
        _interaction_block("Q-learning effect by GAN", q_rows, "gan", thresholds),
        _interaction_block("Q-learning effect by model", q_rows, "model", thresholds),
        _interaction_block("GAN effect by selection", g_rows, "selection", thresholds),
        _interaction_block("GAN effect by model", g_rows, "model", thresholds),
        _interaction_block("LSTM vs Lasso by selection", model_pair_rows, "selection", thresholds),
        _interaction_block("LSTM vs Lasso by GAN", model_pair_rows, "gan", thresholds),
    ]
    interaction_depends = any(block["depends_on_other_factor"] for block in interactions)
    interaction_effect = (
        "produced_a_mixed_effect" if interaction_depends else (
            next((block["effect"] for block in interactions if block["effect"] != "had_minimal_effect"), "had_minimal_effect")
        )
    )

    conditions = []
    for exp_id, row in (results or {}).items():
        overall = row.get("overall") or {}
        conditions.append({
            "condition": exp_id,
            "selection": row.get("selection"),
            "gan": row.get("gan"),
            "model": row.get("model"),
            "mae": _f(overall.get("mae")),
            "r2": _f(overall.get("r2")),
            "official_f1": _f(overall.get("official_f1")),
            "mae_label": _label_mae(_f(overall.get("mae")), thresholds),
        })
    observed_lowest_mae = None
    scored = [c for c in conditions if c["mae"] is not None]
    if scored:
        observed_lowest_mae = min(scored, key=lambda c: c["mae"])["condition"]

    small = n_test is not None and n_test < thresholds.small_test_n
    qualification = (
        f"Comparisons use this run's held-out set"
        + (f" (n_test={n_test})" if n_test is not None else "")
        + ". They are descriptive matched-pair observations, not a statistical "
          "claim that one component is generally better."
    )
    if small:
        qualification += (
            f" n_test is below {thresholds.small_test_n}, so component rankings "
            "from this run are unstable and must not be treated as a result."
        )

    def _effect_block(component: str, label: str, rows: List[Dict[str, Any]], mean_key: str) -> Dict[str, Any]:
        mean = findings.get(mean_key) or {}
        return {
            "component": component,
            "effect": label,
            "mean_delta_mae": _f(mean.get("delta_mae")),
            "mean_delta_r2": _f(mean.get("delta_r2")),
            "matched_pairs": [
                {
                    **{k: row.get(k) for k in row},
                    "effect": _effect_from_delta_mae(row.get("delta_mae"), row.get("delta_r2"), thresholds),
                }
                for row in rows
            ],
            "statement": (
                f"On this run, {component} {label.replace('_', ' ')} relative to its "
                f"matched control on the recorded deltas. {qualification}"
            ),
        }

    model_statement = (
        f"Across the four matched (selection × GAN) cells, Lasso mean MAE="
        f"{lasso_mae if lasso_mae is None else round(lasso_mae, 4)} and LSTM mean MAE="
        f"{lstm_mae if lstm_mae is None else round(lstm_mae, 4)}. "
        f"Relative to Lasso, LSTM {model_label.replace('_', ' ')} on those means. "
        f"{qualification}"
    )

    return {
        "kind": "experiment_effect_interpretation",
        "qlearning_vs_baseline": _effect_block("Q-learning selection", q_label, q_rows, "qlearning_effect_mean"),
        "gan_vs_no_gan": _effect_block("GAN augmentation", g_label, g_rows, "gan_effect_mean"),
        "lstm_vs_lasso": {
            "component": "LSTM vs Lasso",
            "effect": model_label,
            "lasso_mean_mae": lasso_mae,
            "lstm_mean_mae": lstm_mae,
            "lasso_mean_r2": lasso_r2,
            "lstm_mean_r2": lstm_r2,
            "matched_pairs": [
                {
                    **row,
                    "effect": _effect_from_delta_mae(row.get("delta_mae"), row.get("delta_r2"), thresholds),
                }
                for row in model_pair_rows
            ],
            "statement": model_statement,
        },
        "interaction_effects": {
            "effect": interaction_effect,
            "depends_on_other_factor": interaction_depends,
            "comparisons": interactions,
            "statement": (
                f"Recorded component deltas {'varied with the other factor' if interaction_depends else 'did not change category across the other factor'}. "
                f"{qualification}"
            ),
        },
        "factorial_conditions": conditions,
        "lowest_observed_test_mae_condition": observed_lowest_mae,
        "small_test_set": small,
        "qualification": qualification,
        "effect_vocabulary": list(EFFECT_LABELS),
        "note": (
            "improved / reduced refer to recorded test MAE (and R²) deltas on "
            "this run. They are not endorsements of a production model."
        ),
    }


# ---------------------------------------------------------------------------
# Confidence (read-only; never recomputes the heuristic)
# ---------------------------------------------------------------------------

def interpret_confidence(
    confidence_block: Optional[Mapping[str, Any]] = None,
    *,
    heuristic_score: Optional[float] = None,
    n_test: Optional[int] = None,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Pass through metrics-engine confidence semantics. Does not recompute the score."""
    block = confidence_block or {}
    semantics = block.get("semantics") or {
        "name": "model_confidence_heuristic",
        "kind": "certainty_heuristic",
        "calibrated_probability": False,
        "statistical_uncertainty": False,
        "confidence_interval": None,
        "description": (
            "If present, this score was produced by metrics_engine.compute_model_confidence "
            "and is a certainty heuristic, not calibrated statistical uncertainty."
        ),
    }
    lasso_score = heuristic_score
    lstm_score = None
    if isinstance(block.get("lasso"), Mapping):
        lasso_score = _f((block["lasso"].get("aggregate") or {}).get("score"))
    if isinstance(block.get("lstm"), Mapping):
        lstm_score = _f((block["lstm"].get("aggregate") or {}).get("score"))

    small = n_test is not None and n_test < thresholds.small_test_n
    statement = (
        "Model confidence, when present, is a certainty heuristic from the "
        "metrics engine (score distance from 0.5, optionally blended with MAE). "
        "It is not a calibrated probability and not a confidence interval."
    )
    if lasso_score is not None or lstm_score is not None:
        statement += (
            f" Recorded heuristic scores: Lasso={lasso_score}, LSTM={lstm_score}."
        )
    else:
        statement += " No heuristic confidence score was present in the supplied metrics."
    if small:
        statement += " Small held-out n further limits how much weight this score can carry."

    return {
        "kind": "confidence_interpretation",
        "heuristic_scores": {"lasso": lasso_score, "lstm": lstm_score},
        "semantics": semantics,
        "calibrated_probability": False,
        "statistical_uncertainty": False,
        "confidence_interval": None,
        "small_test_set": small,
        "statement": statement,
    }


# ---------------------------------------------------------------------------
# Single-condition interpretation (one cell of the 2×2×2)
# ---------------------------------------------------------------------------

def interpret_condition_result(
    result: Mapping[str, Any],
    *,
    predictions: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    data_quality: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """
    Interpret one completed condition from already-computed runner outputs.

    Does not recalculate metrics. Component comparisons are deferred to the
    experiment-level interpretation after all eight cells exist.
    """
    overall = result.get("overall") or {}
    per_trait = result.get("per_trait") or {}
    reproducibility = result.get("reproducibility") or {}
    n_test = _i(result.get("n_test") or reproducibility.get("test_participant_count"))
    n_train = _i(result.get("n_train") or reproducibility.get("train_participant_count"))
    n_val = _i(result.get("n_val") or reproducibility.get("validation_participant_count"))
    n_users = _i(result.get("n_users") or reproducibility.get("participant_sample_size"))

    regression = {
        "aggregate": {
            "mae": overall.get("mae"),
            "r2": overall.get("r2"),
        },
        "per_trait": {
            key: {"mae": (block or {}).get("mae"), "r2": (block or {}).get("r2")}
            for key, block in per_trait.items()
            if isinstance(block, Mapping)
        },
    }
    binary = {
        "aggregate": {
            "f1": overall.get("official_f1"),
            "roc_auc": overall.get("roc_auc"),
        },
    }
    quality = interpret_prediction_quality(
        regression,
        binary,
        source="condition_overall_and_official_binary",
        thresholds=thresholds,
    )
    high_low = interpret_prediction_quality(
        None,
        {
            "aggregate": {
                "f1": overall.get("official_f1"),
                "roc_auc": overall.get("roc_auc"),
            },
        },
        f1=overall.get("official_f1"),
        roc_auc=overall.get("roc_auc"),
        source="official_high_low_from_validation_threshold",
        thresholds=thresholds,
    )
    evidence = interpret_evidence_quality(
        n_participants=n_users,
        n_train=n_train,
        n_test=n_test,
        mean_comments=_f(result.get("mean_comments_selected")),
        data_quality=data_quality,
        thresholds=thresholds,
        population="condition_shared_split",
    )
    confidence = interpret_confidence(n_test=n_test, thresholds=thresholds)
    prediction = (
        interpret_ocean_predictions(predictions, thresholds)
        if predictions is not None
        else None
    )

    label = result.get("label") or result.get("experiment")
    what_happened = (
        f"Condition {result.get('experiment')} ({label}) ran selection="
        f"{result.get('selection')}, gan={result.get('gan')}, model="
        f"{result.get('model')} on train={n_train}, validation={n_val}, "
        f"test={n_test} participants. Metrics were produced by the metrics engine; "
        "this record only describes those stored values."
    )
    component_note = (
        "This record is one cell of the 2×2×2 matrix. Whether Q-learning, GAN, "
        "or the model family improved its matched control is answered after all "
        "eight conditions are compared."
    )
    limitations = [
        "Predictions are model outputs on a normalized [0, 1] scale, not diagnoses.",
        "High/Low metrics use the validation-selected threshold applied once to test.",
        component_note,
    ]
    if evidence.get("small_test_set"):
        limitations.append(
            f"Held-out n_test={n_test} is below {thresholds.small_test_n}; "
            "do not treat this cell as a stable ranking."
        )

    return {
        "kind": "condition_interpretation",
        "schema_version": 1,
        "uses_llm": False,
        "recalculates_metrics": False,
        "clinical_diagnosis": False,
        "condition": result.get("experiment"),
        "label": label,
        "selection": result.get("selection"),
        "gan": result.get("gan"),
        "model": result.get("model"),
        "questions": {
            "what_happened": what_happened,
            "how_good_was_the_prediction": quality["statement"],
            "how_confident_or_uncertain": confidence["statement"],
            "how_much_usable_evidence": evidence["statement"],
            "did_the_experimental_component_improve_the_baseline": component_note,
        },
        "prediction": prediction,
        "quality": quality,
        "high_low": {
            "source": "official_binary_validation_threshold",
            "official_f1": _f(overall.get("official_f1")),
            "official_accuracy": _f(overall.get("official_accuracy")),
            "roc_auc": _f(overall.get("roc_auc")),
            "pr_auc": _f(overall.get("pr_auc")),
            "labels": high_low["labels"],
            "statement": high_low["statement"],
        },
        "evidence": evidence,
        "confidence": confidence,
        "reproducibility": {
            "repository_commit_sha": reproducibility.get("repository_commit_sha"),
            "repository_dirty": reproducibility.get("repository_dirty"),
            "code_identity_reliable": reproducibility.get("code_identity_reliable"),
            "random_seed": reproducibility.get("random_seed") or result.get("seed"),
            "reproducibility_class": reproducibility.get("reproducibility_class"),
        },
        "config": config or {},
        "limitations": limitations,
        "thresholds_used": asdict(thresholds),
    }


def build_research_questions(
    summary: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    """Machine-readable answers used by the runner's research-question artifact."""
    data_quality = bundle.get("data_quality") or {}
    reproducibility = bundle.get("reproducibility") or {}
    repo = reproducibility.get("repository") or {}
    filt = data_quality.get("experiment_filter") or {}
    retention = data_quality.get("retention") or {}
    ingestion = data_quality.get("ingestion") or {}
    cleaning = data_quality.get("cleaning") or {}
    effects = summary.get("component_effects") or {}
    evidence = (summary.get("confidence_and_evidence") or {}).get("evidence") or {}
    confidence = (summary.get("confidence_and_evidence") or {}).get("confidence") or {}
    performance = summary.get("model_performance") or {}
    lasso_q = performance.get("lasso") or {}
    lstm_q = performance.get("lstm") or {}
    config = bundle.get("config") or {}
    sample = bundle.get("sample") or {}
    q = effects.get("qlearning_vs_baseline") or {}
    g = effects.get("gan_vs_no_gan") or {}
    m = effects.get("lstm_vs_lasso") or {}
    inter = effects.get("interaction_effects") or {}
    limitations = list(summary.get("limitations") or [])

    entered = (
        f"Prepared users before the experiment filter: "
        f"{filt.get('users_before_filter')}. "
        f"Ingestion available={ingestion.get('available')}; "
        f"cleaning available={(cleaning or {}).get('available')}."
    )
    survived = (
        f"{filt.get('users_used')} sampled participants were used "
        f"(eligible={filt.get('users_after_filter')}, "
        f"excluded={filt.get('users_excluded')}). "
        f"Experiment user retention={retention.get('users_experiment')}."
    )
    configuration = (
        f"2×2×2 factorial on seed={config.get('seed')}: "
        f"selection × GAN × model, n_users={sample.get('n_users')}, "
        f"n_train={sample.get('n_train')}, n_val={sample.get('n_val')}, "
        f"n_test={sample.get('n_test')}."
    )
    sha = repo.get("repository_commit_sha") or (summary.get("experiment_configuration") or {}).get(
        "repository_commit_sha"
    )
    dirty = repo.get("repository_dirty")
    code_version = (
        f"Repository commit {sha}"
        + (" (working tree dirty; code identity is not reliable)." if dirty else ".")
    )
    ocean = (
        f"Continuous OCEAN (read metrics, not recomputed): "
        f"Lasso {lasso_q.get('statement')}; LSTM {lstm_q.get('statement')}."
    )
    high_low_f1 = (lasso_q.get("metrics_read") or {}).get("f1")
    high_low = (
        "High/Low uses the validation-selected threshold applied once to test. "
        f"Recorded official F1 (headline/Lasso read)={high_low_f1}; "
        f"ROC-AUC={(lasso_q.get('metrics_read') or {}).get('roc_auc')}."
    )
    q_help = (
        f"Q-learning {str(q.get('effect') or 'had_minimal_effect').replace('_', ' ')} "
        f"versus baseline-select on this run's matched pairs."
    )
    g_help = (
        f"GAN {str(g.get('effect') or 'had_minimal_effect').replace('_', ' ')} "
        f"versus no-GAN on this run's matched pairs."
    )
    model_help = (
        f"Relative to Lasso, LSTM {str(m.get('effect') or 'had_minimal_effect').replace('_', ' ')} "
        "on the recorded matched-cell means. "
        f"{effects.get('qualification') or ''}"
    ).strip()
    small = bool(evidence.get("small_test_set") or effects.get("small_test_set"))
    stability = (
        f"{confidence.get('statement')} "
        f"Interaction observation: {inter.get('effect')}. "
        + (
            "Small held-out n means rankings from this run are unstable."
            if small
            else "Stability is still limited to this sample and seed; a matching seed is not statistical replication."
        )
    )
    return {
        "what_data_entered_the_pipeline": entered,
        "what_data_survived": survived,
        "which_configuration_was_run": configuration,
        "which_code_version_ran_it": code_version,
        "how_well_did_it_predict_ocean": ocean,
        "how_did_it_perform_under_high_low_interpretation": high_low,
        "did_qlearning_help": q_help,
        "did_gan_help": g_help,
        "did_lasso_or_lstm_perform_better": model_help,
        "how_stable_is_the_result": stability,
        "what_limitations_apply": limitations,
    }


# ---------------------------------------------------------------------------
# E. Research summary + five questions
# ---------------------------------------------------------------------------

def interpret_research_summary(
    bundle: Mapping[str, Any],
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
    *,
    predictions: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    evaluate_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured summary suitable for a thesis table or research write-up."""
    sample = bundle.get("sample") or {}
    config = bundle.get("config") or {}
    findings = bundle.get("findings") or {}
    data_quality = bundle.get("data_quality") or {}
    reproducibility = bundle.get("reproducibility") or {}
    results = bundle.get("results") or {}

    n_test = _i(sample.get("n_test"))
    n_train = _i(sample.get("n_train"))
    n_val = _i(sample.get("n_val"))
    n_users = _i(sample.get("n_users"))

    hybrid = bundle.get("hybrid_cell_evaluations") or {}
    if evaluate_payload is None and hybrid:
        evaluate_payload = next(iter(hybrid.values()), None)
    evaluate_payload = evaluate_payload or {}

    regression = evaluate_payload.get("continuous_regression") or evaluate_payload.get("lasso")
    if isinstance(regression, Mapping) and "lasso" in regression:
        # Prefer Lasso block for a headline continuous description; both are stored below.
        lasso_reg = regression.get("lasso")
        lstm_reg = regression.get("lstm")
    else:
        lasso_reg = evaluate_payload.get("lasso")
        lstm_reg = evaluate_payload.get("lstm")

    binary = evaluate_payload.get("binary_interpretation") or {}
    confidence_block = evaluate_payload.get("confidence")

    # Fallback to runner overall if evaluate() payload is absent.
    headline_overall = None
    if findings.get("best_condition") and results:
        best_id = findings["best_condition"].get("condition")
        headline_overall = (results.get(best_id) or {}).get("overall")
    if headline_overall is None and results:
        headline_overall = next(iter(results.values()), {}).get("overall")

    quality_lasso = interpret_prediction_quality(
        lasso_reg or headline_overall,
        (binary.get("lasso") if isinstance(binary, Mapping) else None) or headline_overall,
        source="metrics_engine_or_runner_overall",
        thresholds=thresholds,
    )
    quality_lstm = interpret_prediction_quality(
        lstm_reg,
        binary.get("lstm") if isinstance(binary, Mapping) else None,
        source="metrics_engine_or_runner_overall",
        thresholds=thresholds,
    )
    evidence = interpret_evidence_quality(
        n_participants=n_users,
        n_train=n_train,
        n_test=n_test,
        data_quality=data_quality,
        thresholds=thresholds,
        population="experiment_sample",
    )
    effects = interpret_experiment_effects(bundle, n_test=n_test, thresholds=thresholds)
    confidence = interpret_confidence(confidence_block, n_test=n_test, thresholds=thresholds)
    prediction = (
        interpret_ocean_predictions(predictions, thresholds)
        if predictions is not None
        else None
    )

    what_happened = {
        "design": "2×2×2 factorial: selection (baseline vs Q-learning) × GAN × model (Lasso vs LSTM)",
        "n_conditions": len(results),
        "n_users": n_users,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "label_scale": sample.get("label_scale"),
        "dataset_source": (reproducibility.get("dataset_source") or config.get("dataset_source")),
        "statement": (
            f"This run evaluated {len(results)} factorial conditions on "
            f"{n_users if n_users is not None else 'an unstated number of'} "
            f"participants (train={n_train}, validation={n_val}, test={n_test}) "
            f"using continuous 5-output OCEAN regression for both Lasso and LSTM."
        ),
    }

    limitations = [
        "Predictions are model outputs on a normalized [0, 1] scale, not clinical or psychological diagnoses.",
        "Model confidence, when present, is a certainty heuristic and not calibrated statistical uncertainty.",
        "Quality and evidence labels use project reporting bands, not external published cutoffs.",
        "Component effects are matched-pair observations from this run only.",
    ]
    if evidence.get("small_test_set"):
        limitations.append(
            f"Held-out n_test={n_test} is below {thresholds.small_test_n}; "
            "do not treat component rankings as stable."
        )
    if (data_quality.get("ingestion") or {}).get("available") is False:
        limitations.append(
            "Ingestion-level data-quality counts were not measured in this process."
        )
    grouping = ((data_quality.get("ingestion") or {}).get("group_by")
                or ((reproducibility.get("preprocessing") or {}).get("group_by")))
    if grouping:
        limitations.append(
            f"Participants are proxy users grouped by '{grouping}', not verified author IDs."
        )

    answers = {
        "what_happened": what_happened["statement"],
        "how_good_was_the_prediction": quality_lasso["statement"],
        "how_confident_or_uncertain": confidence["statement"],
        "how_much_usable_evidence": evidence["statement"],
        "did_the_experimental_component_improve_the_baseline": (
            f"Q-learning {effects['qlearning_vs_baseline']['effect'].replace('_', ' ')}; "
            f"GAN {effects['gan_vs_no_gan']['effect'].replace('_', ' ')}; "
            f"LSTM vs Lasso {effects['lstm_vs_lasso']['effect'].replace('_', ' ')}; "
            f"interactions {effects['interaction_effects']['effect'].replace('_', ' ')}. "
            f"{effects['qualification']}"
        ),
    }

    thesis_rows = [
        {"section": "configuration", "metric": "n_users", "value": n_users},
        {"section": "configuration", "metric": "n_train", "value": n_train},
        {"section": "configuration", "metric": "n_val", "value": n_val},
        {"section": "configuration", "metric": "n_test", "value": n_test},
        {"section": "quality", "metric": "lasso_continuous_label", "value": quality_lasso["labels"]["continuous"]},
        {"section": "quality", "metric": "lstm_continuous_label", "value": quality_lstm["labels"]["continuous"]},
        {"section": "quality", "metric": "lasso_mae_read", "value": quality_lasso["metrics_read"]["mae"]},
        {"section": "quality", "metric": "lstm_mae_read", "value": quality_lstm["metrics_read"]["mae"]},
        {"section": "evidence", "metric": "overall_evidence_label", "value": evidence["labels"]["overall"]},
        {"section": "evidence", "metric": "small_test_set", "value": evidence["small_test_set"]},
        {"section": "effects", "metric": "qlearning_vs_baseline", "value": effects["qlearning_vs_baseline"]["effect"]},
        {"section": "effects", "metric": "gan_vs_no_gan", "value": effects["gan_vs_no_gan"]["effect"]},
        {"section": "effects", "metric": "lstm_vs_lasso", "value": effects["lstm_vs_lasso"]["effect"]},
        {"section": "effects", "metric": "interaction_effects", "value": effects["interaction_effects"]["effect"]},
        {
            "section": "effects",
            "metric": "lowest_observed_test_mae_condition",
            "value": effects["lowest_observed_test_mae_condition"],
        },
        {"section": "confidence", "metric": "calibrated_probability", "value": False},
    ]

    payload = {
        "kind": "research_summary",
        "schema_version": 1,
        "experiment_configuration": {
            "config": config,
            "sample": {
                "n_users": n_users,
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "label_scale": sample.get("label_scale"),
            },
            "repository_commit_sha": (reproducibility.get("repository") or {}).get(
                "repository_commit_sha"
            ),
        },
        "data_quality": {
            "users_used": (data_quality.get("experiment_filter") or {}).get("users_used"),
            "users_excluded": (data_quality.get("experiment_filter") or {}).get("users_excluded"),
            "retention": data_quality.get("retention"),
            "comment_volume_train": (data_quality.get("comment_volume") or {}).get("train"),
            "comment_volume_test": (data_quality.get("comment_volume") or {}).get("test"),
            "ingestion_available": (data_quality.get("ingestion") or {}).get("available"),
        },
        "model_performance": {
            "lasso": quality_lasso,
            "lstm": quality_lstm,
            "findings_best_condition": findings.get("best_condition"),
        },
        "component_effects": effects,
        "confidence_and_evidence": {
            "confidence": confidence,
            "evidence": evidence,
        },
        "prediction": prediction,
        "limitations": limitations,
        "answers": answers,
        "thesis_rows": thesis_rows,
        "thresholds_used": asdict(thresholds),
    }
    payload["research_questions"] = build_research_questions(payload, bundle)
    return payload


def interpret_experiment_bundle(
    bundle: Mapping[str, Any],
    thresholds: InterpretationThresholds = DEFAULT_THRESHOLDS,
    *,
    predictions: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    evaluate_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single entry point: answer the five questions from a runner bundle.

    Machine-readable. Deterministic for the same bundle and thresholds.
    """
    summary = interpret_research_summary(
        bundle,
        thresholds,
        predictions=predictions,
        evaluate_payload=evaluate_payload,
    )
    return {
        "kind": "deterministic_interpretation",
        "schema_version": 1,
        "uses_llm": False,
        "recalculates_metrics": False,
        "clinical_diagnosis": False,
        "questions": {
            "what_happened": summary["answers"]["what_happened"],
            "how_good_was_the_prediction": summary["answers"]["how_good_was_the_prediction"],
            "how_confident_or_uncertain": summary["answers"]["how_confident_or_uncertain"],
            "how_much_usable_evidence": summary["answers"]["how_much_usable_evidence"],
            "did_the_experimental_component_improve_the_baseline": summary["answers"][
                "did_the_experimental_component_improve_the_baseline"
            ],
        },
        "prediction": summary.get("prediction"),
        "quality": summary["model_performance"],
        "evidence": summary["confidence_and_evidence"]["evidence"],
        "confidence": summary["confidence_and_evidence"]["confidence"],
        "effects": summary["component_effects"],
        "research_questions": summary.get("research_questions"),
        "research_summary": summary,
        "thresholds_used": asdict(thresholds),
    }
