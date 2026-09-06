"""
Research-readiness gate for the PANDORA factorial experiment.

Validates recorded artifacts (and, when present, runner source) against the
research contract. Does not retrain models, recalculate metrics, or change
architecture.

Statuses
--------
PASS     required evidence is present and valid
WARNING  incomplete or qualified; does not stop a run
FAIL     missing or invalid condition; the report names the exact gap

A WARNING never stops execution. The CLI exits 1 only on FAIL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

EXPECTED_CONDITIONS: Tuple[str, ...] = (
    "lasso_baseline",
    "lasso_qlearn",
    "lasso_baseline_gan",
    "lasso_qlearn_gan",
    "lstm_baseline",
    "lstm_qlearn",
    "lstm_baseline_gan",
    "lstm_qlearn_gan",
)
HEADLINE_REGRESSION = ("mae", "rmse", "r2")
HEADLINE_REGRESSION_ALIASES = {
    "mae": ("mae", "val_mae"),
    "rmse": ("rmse", "val_rmse"),
    "r2": ("r2", "val_r2"),
}
HEADLINE_BINARY = ("official_f1", "official_accuracy")
HEADLINE_BINARY_ALIASES = ("official_f1", "macro_f1", "f1", "official_accuracy", "accuracy")
HEADLINE_CURVES = ("roc_auc", "pr_auc")
OVERSTATED_CLAIM = re.compile(
    r"\b(statistically significant|calibrated probability|p\s*<\s*0|"
    r"the person is definitely|clinical diagnosis|diagnosed as)\b",
    re.IGNORECASE,
)
NEGATED_CAUTION = re.compile(
    r"\b(not|never|no)\b.{0,40}\b(calibrated|diagnosis|diagnoses|statistically significant|"
    r"confidence interval|generally better)\b",
    re.IGNORECASE,
)

Payload = Union[str, Path, Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Report primitives
# ---------------------------------------------------------------------------

def _check(
    section: str,
    check_id: str,
    status: str,
    detail: str,
    *,
    missing: Optional[str] = None,
    condition: Optional[str] = None,
) -> Dict[str, Any]:
    if status not in {"PASS", "WARNING", "FAIL"}:
        raise ValueError(f"Invalid status {status!r}")
    item = {
        "section": section,
        "id": check_id,
        "status": status,
        "detail": detail,
    }
    if missing:
        item["missing"] = missing
    if condition:
        item["condition"] = condition
    return item


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return False
    return True


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _first_float(block: Mapping[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = _as_float(block.get(name))
        if value is not None:
            return value
    return None


def _records(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []
    if hasattr(obj, "to_dict"):
        try:
            return list(obj.to_dict("records"))
        except Exception:
            return []
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, Mapping)]
    if isinstance(obj, Mapping):
        if any(isinstance(v, (int, float)) for v in obj.values()):
            return [dict(obj)]
    return []


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_strings(value)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_run_payload(source: Payload) -> Dict[str, Any]:
    """Accept a bundle dict, a JSON file, or an experiment output directory."""
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_file():
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} did not contain a JSON object.")
        return data
    if not path.is_dir():
        raise FileNotFoundError(f"No run payload at {path}")

    multi_path = path / "multi_seed_summary.json"
    if multi_path.exists():
        payload = _load_json(multi_path)
        if not isinstance(payload, dict):
            raise ValueError(f"{multi_path} did not contain a JSON object.")
        runs = dict(payload.get("runs") or {})
        for child in sorted(path.glob("seed_*")):
            if not child.is_dir():
                continue
            seed_key = child.name.replace("seed_", "", 1)
            summary = child / "run_summary.json"
            if summary.exists():
                runs[seed_key] = _load_json(summary)
        payload["runs"] = runs
        payload.setdefault("kind", "multi_seed_experiment")
        return payload

    for name in ("run_summary.json", "reproducibility_summary.json", "reproducibility.json"):
        candidate = path / name
        if candidate.exists():
            data = _load_json(candidate)
            if isinstance(data, dict):
                data.setdefault("_artifact_dir", str(path))
                _attach_sidecar_tables(data, path)
                return data
    raise FileNotFoundError(
        f"{path} has no run_summary.json / multi_seed_summary.json."
    )


def _attach_sidecar_tables(bundle: Dict[str, Any], directory: Path) -> None:
    """Recover comparison tables that are stored as CSV, not in run_summary.json."""
    try:
        import csv
    except ImportError:
        return
    mapping = {
        "qlearning_effect": directory / "qlearning_effect.csv",
        "gan_effect": directory / "gan_effect.csv",
    }
    effects = dict(bundle.get("factor_effects") or {})
    for key, csv_path in mapping.items():
        if key in effects or not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            effects[key] = list(csv.DictReader(handle))
    if effects:
        bundle["factor_effects"] = effects
    model_csv = directory / "model_comparison.csv"
    if "model_comparison" not in bundle and model_csv.exists():
        with model_csv.open("r", encoding="utf-8", newline="") as handle:
            import csv
            bundle["model_comparison"] = list(csv.DictReader(handle))
    interp_path = directory / "interpretation.json"
    if "interpretation" not in bundle and interp_path.exists():
        bundle["interpretation"] = _load_json(interp_path)
    dq_path = directory / "data_quality.json"
    if "data_quality" not in bundle and dq_path.exists():
        bundle["data_quality"] = _load_json(dq_path)


# ---------------------------------------------------------------------------
# Section checks
# ---------------------------------------------------------------------------

def _reproducibility_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "REPRODUCIBILITY"
    config = bundle.get("config") or {}
    sample = bundle.get("sample") or {}
    repro = bundle.get("reproducibility") or {}
    summary = bundle.get("reproducibility_summary") or {}
    repo = repro.get("repository") or {}
    out: List[Dict[str, Any]] = []

    seed = (
        repro.get("random_seed")
        or summary.get("random_seed")
        or sample.get("seed")
        or config.get("seed")
    )
    if _present(seed):
        out.append(_check(section, "reproducibility.seed_recorded", "PASS", f"Seed recorded ({seed})."))
    else:
        out.append(_check(
            section, "reproducibility.seed_recorded", "FAIL",
            "No random seed was recorded.",
            missing="reproducibility.random_seed / config.seed",
        ))

    sha = (
        repo.get("repository_commit_sha")
        or summary.get("repository_commit_sha")
        or (bundle.get("interpretation") or {}).get("research_summary", {})
        .get("experiment_configuration", {})
        .get("repository_commit_sha")
    )
    if _present(sha):
        dirty = repo.get("repository_dirty") or summary.get("repository_dirty")
        if dirty or summary.get("code_identity_reliable") is False:
            out.append(_check(
                section, "reproducibility.commit_sha_recorded", "WARNING",
                f"Commit SHA recorded ({sha}) but the working tree is dirty or code identity is not reliable.",
            ))
        else:
            out.append(_check(section, "reproducibility.commit_sha_recorded", "PASS", f"Commit SHA recorded ({sha})."))
    else:
        out.append(_check(
            section, "reproducibility.commit_sha_recorded", "FAIL",
            "No repository commit SHA was recorded.",
            missing="reproducibility.repository.repository_commit_sha",
        ))

    fingerprint = repro.get("config_fingerprint") or summary.get("config_fingerprint")
    if _present(config) or _present(fingerprint):
        out.append(_check(
            section, "reproducibility.configuration_recorded", "PASS",
            "Experiment configuration is recorded"
            + (f" (fingerprint {fingerprint})." if fingerprint else "."),
        ))
    else:
        out.append(_check(
            section, "reproducibility.configuration_recorded", "FAIL",
            "No experiment configuration or config fingerprint was recorded.",
            missing="config / reproducibility.config_fingerprint",
        ))

    source = (
        repro.get("dataset_source")
        or summary.get("dataset_source")
        or config.get("dataset_source")
        or sample.get("dataset_source")
    )
    if _present(source):
        out.append(_check(section, "reproducibility.dataset_source_recorded", "PASS", f"Dataset/source recorded ({source})."))
    else:
        out.append(_check(
            section, "reproducibility.dataset_source_recorded", "FAIL",
            "No dataset/source identifier was recorded.",
            missing="reproducibility.dataset_source / config.dataset_source",
        ))

    n_train = _as_int(
        sample.get("n_train") or repro.get("train_participant_count") or summary.get("train_participant_count")
    )
    n_val = _as_int(
        sample.get("n_val") or repro.get("validation_participant_count") or summary.get("validation_participant_count")
    )
    n_test = _as_int(
        sample.get("n_test")
        or repro.get("test_participant_count")
        or summary.get("test_participant_count")
        or sample.get("n_val")
        or repro.get("validation_participant_count")
    )
    if n_train is not None and n_val is not None and n_test is not None:
        if n_train < 1 or n_val < 1 or n_test < 1:
            out.append(_check(
                section, "reproducibility.fold_counts_recorded", "FAIL",
                f"Fold counts are recorded but a fold is empty (train={n_train}, val={n_val}, test={n_test}).",
                missing="non-empty train/validation/test counts",
            ))
        else:
            out.append(_check(
                section, "reproducibility.fold_counts_recorded", "PASS",
                f"Train/validation/test counts recorded (train={n_train}, val={n_val}, test={n_test}).",
            ))
    else:
        missing = []
        if n_train is None:
            missing.append("n_train")
        if n_val is None:
            missing.append("n_val")
        if n_test is None:
            missing.append("n_test")
        out.append(_check(
            section, "reproducibility.fold_counts_recorded", "FAIL",
            "Train/validation/test participant counts are incomplete.",
            missing=", ".join(missing),
        ))
    return out


def _data_quality_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "DATA QUALITY"
    dq = bundle.get("data_quality") or {}
    filt = dq.get("experiment_filter") or {}
    cleaning = dq.get("cleaning") or {}
    ingestion = dq.get("ingestion") or {}
    retention = dq.get("retention") or {}
    out: List[Dict[str, Any]] = []

    input_count = (
        _as_int(filt.get("users_before_filter"))
        or _as_int(ingestion.get("comments_before_cleaning"))
        or _as_int(ingestion.get("users_before_cleaning"))
        or _as_int(cleaning.get("comments_before_cleaning"))
    )
    if input_count is not None:
        out.append(_check(section, "data_quality.input_count_recorded", "PASS", f"Input count recorded ({input_count})."))
    else:
        out.append(_check(
            section, "data_quality.input_count_recorded", "FAIL",
            "No input count was recorded (users or comments before filtering/cleaning).",
            missing="data_quality.experiment_filter.users_before_filter / ingestion.comments_before_cleaning",
        ))

    cleaned_count = (
        _as_int(cleaning.get("comments_after_cleaning"))
        or _as_int(ingestion.get("comments_after_cleaning"))
        or _as_int(filt.get("users_used"))
        or _as_int(filt.get("users_after_filter"))
    )
    if cleaned_count is not None:
        if cleaning.get("available") is False and ingestion.get("available") is False:
            out.append(_check(
                section, "data_quality.cleaned_count_recorded", "WARNING",
                f"Experiment-level used/eligible count is recorded ({cleaned_count}), "
                "but ingestion/cleaning counts were not measured in this process.",
            ))
        else:
            out.append(_check(
                section, "data_quality.cleaned_count_recorded", "PASS",
                f"Cleaned/used count recorded ({cleaned_count}).",
            ))
    else:
        out.append(_check(
            section, "data_quality.cleaned_count_recorded", "FAIL",
            "No cleaned or used count was recorded.",
            missing="data_quality.cleaning.comments_after_cleaning / experiment_filter.users_used",
        ))

    percent = None
    for candidate in (
        (retention.get("users_experiment") or {}).get("percent") if isinstance(retention.get("users_experiment"), Mapping) else retention.get("users_experiment"),
        (retention.get("comments_cleaning") or {}).get("percent") if isinstance(retention.get("comments_cleaning"), Mapping) else retention.get("comments_cleaning"),
        (filt.get("user_retention") or {}).get("percent") if isinstance(filt.get("user_retention"), Mapping) else None,
        (ingestion.get("comment_retention") or {}).get("percent") if isinstance(ingestion.get("comment_retention"), Mapping) else None,
    ):
        percent = _as_float(candidate)
        if percent is not None:
            break
    if percent is not None:
        out.append(_check(section, "data_quality.retention_percentage_available", "PASS", f"Retention percentage available ({percent})."))
    else:
        den = None
        if isinstance(filt.get("user_retention"), Mapping):
            den = filt["user_retention"].get("denominator")
        if den == 0:
            out.append(_check(
                section, "data_quality.retention_percentage_available", "WARNING",
                "Retention rate is undefined because the denominator is zero.",
            ))
        else:
            out.append(_check(
                section, "data_quality.retention_percentage_available", "FAIL",
                "No retention percentage is available.",
                missing="data_quality.retention.*.percent / experiment_filter.user_retention.percent",
            ))

    reasons = dq.get("exclusion_reasons") or filt.get("exclusion_reasons")
    if isinstance(reasons, Mapping) and reasons:
        out.append(_check(
            section, "data_quality.exclusion_reasons_available", "PASS",
            f"Exclusion reasons available ({len(reasons)} recorded).",
        ))
    else:
        out.append(_check(
            section, "data_quality.exclusion_reasons_available", "FAIL",
            "No exclusion reasons were recorded.",
            missing="data_quality.exclusion_reasons / experiment_filter.exclusion_reasons",
        ))
    return out


def _results_map(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    results = bundle.get("results") or {}
    return results if isinstance(results, Mapping) else {}


def _prediction_quality_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "PREDICTION QUALITY"
    results = _results_map(bundle)
    out: List[Dict[str, Any]] = []
    if not results:
        out.append(_check(
            section, "prediction_quality.results_present", "FAIL",
            "No per-condition results were present to inspect metrics.",
            missing="results",
        ))
        return out

    for exp_id, row in results.items():
        overall = (row or {}).get("overall") or {}
        per_trait = (row or {}).get("per_trait") or {}
        thresholds = (row or {}).get("threshold_selection") or {}

        model = (row or {}).get("model")
        for metric in HEADLINE_REGRESSION:
            aliases = HEADLINE_REGRESSION_ALIASES.get(metric, (metric,))
            value = _first_float(overall, *aliases)
            if value is not None:
                out.append(_check(
                    section, f"prediction_quality.{metric}", "PASS",
                    f"{metric.upper()} available ({value}).",
                    condition=exp_id,
                ))
            elif model == "lstm":
                out.append(_check(
                    section, f"prediction_quality.{metric}", "WARNING",
                    f"{metric.upper()} is not the official LSTM metric on this runner "
                    "(Low/High classification is).",
                    condition=exp_id,
                ))
            else:
                out.append(_check(
                    section, f"prediction_quality.{metric}", "FAIL",
                    f"{metric.upper()} is missing or non-numeric.",
                    missing=f"results.{exp_id}.overall.{metric}|val_{metric}",
                    condition=exp_id,
                ))

        binary_ok = any(_as_float(overall.get(key)) is not None for key in HEADLINE_BINARY_ALIASES)
        if binary_ok:
            out.append(_check(
                section, "prediction_quality.classification_metrics", "PASS",
                "Official classification metrics are available (F1/accuracy or aliases).",
                condition=exp_id,
            ))
        else:
            out.append(_check(
                section, "prediction_quality.classification_metrics", "FAIL",
                "Classification metrics (official F1/accuracy or macro_f1/accuracy) are missing.",
                missing=f"results.{exp_id}.overall.official_f1|macro_f1|accuracy",
                condition=exp_id,
            ))

        recorded_tau = None
        per_trait_thr = thresholds.get("per_trait") if isinstance(thresholds, Mapping) else None
        if isinstance(per_trait_thr, Mapping):
            recorded_tau = next(
                (
                    block.get("threshold")
                    for block in per_trait_thr.values()
                    if isinstance(block, Mapping) and block.get("threshold") is not None
                ),
                None,
            )
        if recorded_tau is None and isinstance(per_trait, Mapping):
            for block in per_trait.values():
                if not isinstance(block, Mapping):
                    continue
                sweep = block.get("threshold_sweep")
                recorded_tau = (
                    block.get("best_threshold")
                    or (block.get("official_binary") or {}).get("threshold")
                    or (sweep.get("best_threshold") if isinstance(sweep, Mapping) else None)
                )
                if recorded_tau is not None:
                    break
        if recorded_tau is not None:
            source = (thresholds.get("split") if isinstance(thresholds, Mapping) else None) or "validation"
            out.append(_check(
                section, "prediction_quality.threshold_recorded", "PASS",
                f"Decision threshold recorded ({recorded_tau}, source={source}).",
                condition=exp_id,
            ))
        else:
            out.append(_check(
                section, "prediction_quality.threshold_recorded", "FAIL",
                "No validation-selected decision threshold was recorded.",
                missing=f"results.{exp_id}.threshold_selection.per_trait.*.threshold",
                condition=exp_id,
            ))

        for metric, label in (("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")):
            in_trait = (
                isinstance(per_trait, Mapping)
                and any(
                    isinstance(block, Mapping) and _as_float(block.get(metric)) is not None
                    for block in per_trait.values()
                )
            )
            value = _as_float(overall.get(metric))
            if value is not None or in_trait:
                out.append(_check(
                    section, f"prediction_quality.{metric}", "PASS",
                    f"{label} available ({value if value is not None else 'per-trait'}).",
                    condition=exp_id,
                ))
            elif model == "lasso":
                out.append(_check(
                    section, f"prediction_quality.{metric}", "WARNING",
                    f"{label} is not produced by the Lasso tertile/regression path.",
                    condition=exp_id,
                ))
            else:
                out.append(_check(
                    section, f"prediction_quality.{metric}", "FAIL",
                    f"{label} key is absent.",
                    missing=f"results.{exp_id}.overall.{metric}",
                    condition=exp_id,
                ))
    return out


def _interpretation_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "INTERPRETATION"
    interp = bundle.get("interpretation") or {}
    out: List[Dict[str, Any]] = []

    if interp.get("kind") == "interpretation_error":
        out.append(_check(
            section, "interpretation.available", "FAIL",
            f"Interpretation failed: {interp.get('error') or 'unknown error'}.",
            missing="successful interpretation payload",
        ))
        return out
    if not interp:
        # Per-condition sidecars may still exist on results.
        condition_interp = [
            row.get("interpretation")
            for row in _results_map(bundle).values()
            if isinstance(row, Mapping) and row.get("interpretation")
        ]
        if not condition_interp:
            out.append(_check(
                section, "interpretation.available", "FAIL",
                "No experiment-level or per-condition interpretation was recorded.",
                missing="interpretation",
            ))
            return out
        interp = {"condition_interpretations": condition_interp}

    prediction = interp.get("prediction")
    if prediction is None:
        for row in _results_map(bundle).values():
            pred = ((row or {}).get("interpretation") or {}).get("prediction")
            if pred:
                prediction = pred
                break
    if _present(prediction):
        out.append(_check(section, "interpretation.prediction_available", "PASS", "Prediction interpretation is available."))
    else:
        out.append(_check(
            section, "interpretation.prediction_available", "WARNING",
            "No per-trait prediction interpretation was attached (quality/effects may still be present).",
            missing="interpretation.prediction",
        ))

    evidence = interp.get("evidence") or ((interp.get("research_summary") or {}).get("confidence_and_evidence") or {}).get("evidence")
    if not evidence:
        for row in _results_map(bundle).values():
            ev = ((row or {}).get("interpretation") or {}).get("evidence")
            if ev:
                evidence = ev
                break
    if _present(evidence):
        out.append(_check(section, "interpretation.evidence_quality_available", "PASS", "Evidence-quality interpretation is available."))
    else:
        out.append(_check(
            section, "interpretation.evidence_quality_available", "FAIL",
            "Evidence-quality interpretation is missing.",
            missing="interpretation.evidence",
        ))

    confidence = interp.get("confidence") or ((interp.get("research_summary") or {}).get("confidence_and_evidence") or {}).get("confidence") or {}
    overstated = []
    if confidence.get("calibrated_probability") is True:
        overstated.append("confidence.calibrated_probability is True")
    if confidence.get("statistical_uncertainty") is True:
        overstated.append("confidence.statistical_uncertainty is True")
    if interp.get("clinical_diagnosis") is True:
        overstated.append("clinical_diagnosis is True")
    if (bundle.get("paired_differences") or {}).get("significance_claimed") is True:
        overstated.append("paired_differences.significance_claimed is True")
    claim_hits = []
    for text in _walk_strings(interp):
        if OVERSTATED_CLAIM.search(text) and not NEGATED_CAUTION.search(text):
            claim_hits.append(text[:160])
    if overstated or claim_hits:
        out.append(_check(
            section, "interpretation.confidence_not_overstated", "FAIL",
            "Confidence/uncertainty wording is overstated: "
            + "; ".join(overstated or claim_hits[:2]),
            missing="non-overstated confidence language",
        ))
    else:
        out.append(_check(
            section, "interpretation.confidence_not_overstated", "PASS",
            "Confidence wording is not overstated (heuristic only; not calibrated).",
        ))

    effects = interp.get("effects") or ((interp.get("research_summary") or {}).get("component_effects"))
    if isinstance(effects, Mapping) and (
        effects.get("qlearning_vs_baseline")
        or effects.get("gan_vs_no_gan")
        or effects.get("lstm_vs_lasso")
    ):
        out.append(_check(section, "interpretation.experiment_effects_reported", "PASS", "Experiment-effect interpretation is reported."))
    else:
        out.append(_check(
            section, "interpretation.experiment_effects_reported", "FAIL",
            "Experiment-effect interpretation is missing.",
            missing="interpretation.effects.qlearning_vs_baseline / gan_vs_no_gan / lstm_vs_lasso",
        ))
    return out


def _experiment_design_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "EXPERIMENT DESIGN"
    results = _results_map(bundle)
    out: List[Dict[str, Any]] = []
    present = set(results)
    missing = [exp_id for exp_id in EXPECTED_CONDITIONS if exp_id not in present]
    extra = sorted(present - set(EXPECTED_CONDITIONS))
    if not missing and len(present) >= 8:
        out.append(_check(
            section, "experiment_design.all_eight_conditions", "PASS",
            "All 8 factorial conditions are present.",
        ))
    else:
        out.append(_check(
            section, "experiment_design.all_eight_conditions", "FAIL",
            "Expected 8 configurations are not all present."
            + (f" Extra ids: {extra}." if extra else ""),
            missing=", ".join(missing) if missing else "results",
            condition="factorial_matrix",
        ))

    findings = bundle.get("findings") or {}
    effects = bundle.get("factor_effects") or {}
    model_cmp = bundle.get("model_comparison")
    interp_effects = (bundle.get("interpretation") or {}).get("effects") or {}
    paired = bundle.get("paired_differences") or {}

    def _has_q() -> bool:
        if _records(effects.get("qlearning_effect")):
            return True
        if findings.get("qlearning_effect_mean"):
            return True
        if (interp_effects.get("qlearning_vs_baseline") or {}).get("effect"):
            return True
        if paired.get("qlearning_minus_baseline"):
            return True
        return any(
            (row or {}).get("selection") == "baseline" for row in results.values()
        ) and any(
            (row or {}).get("selection") == "qlearning" for row in results.values()
        )

    def _has_gan() -> bool:
        if _records(effects.get("gan_effect")):
            return True
        if findings.get("gan_effect_mean"):
            return True
        if (interp_effects.get("gan_vs_no_gan") or {}).get("effect"):
            return True
        if paired.get("gan_minus_no_gan"):
            return True
        return any(bool((row or {}).get("gan")) for row in results.values()) and any(
            (row or {}).get("gan") is False for row in results.values()
        )

    def _has_model() -> bool:
        if _records(model_cmp):
            return True
        if findings.get("better_model") or findings.get("model_means"):
            return True
        if (interp_effects.get("lstm_vs_lasso") or {}).get("effect"):
            return True
        if paired.get("lstm_minus_lasso"):
            return True
        return any((row or {}).get("model") == "lasso" for row in results.values()) and any(
            (row or {}).get("model") == "lstm" for row in results.values()
        )

    if any((row or {}).get("selection") == "baseline" for row in results.values()) or _has_q():
        out.append(_check(section, "experiment_design.baseline_comparisons", "PASS", "Baseline comparisons exist."))
    else:
        out.append(_check(
            section, "experiment_design.baseline_comparisons", "FAIL",
            "No baseline-select condition or baseline comparison table was found.",
            missing="results[selection=baseline] / factor_effects.qlearning_effect",
        ))

    if _has_q():
        out.append(_check(section, "experiment_design.qlearning_comparisons", "PASS", "Q-learning comparisons exist."))
    else:
        out.append(_check(
            section, "experiment_design.qlearning_comparisons", "FAIL",
            "No Q-learning vs baseline comparison was found.",
            missing="factor_effects.qlearning_effect / findings.qlearning_effect_mean",
        ))

    if _has_gan():
        out.append(_check(section, "experiment_design.gan_comparisons", "PASS", "GAN comparisons exist."))
    else:
        out.append(_check(
            section, "experiment_design.gan_comparisons", "FAIL",
            "No GAN vs no-GAN comparison was found.",
            missing="factor_effects.gan_effect / findings.gan_effect_mean",
        ))

    if _has_model():
        out.append(_check(section, "experiment_design.lasso_lstm_comparisons", "PASS", "Lasso/LSTM comparisons exist."))
    else:
        out.append(_check(
            section, "experiment_design.lasso_lstm_comparisons", "FAIL",
            "No Lasso vs LSTM comparison was found.",
            missing="model_comparison / findings.model_means",
        ))
    return out


def _split_sets(bundle: Mapping[str, Any]) -> Tuple[Optional[set], Optional[set], Optional[set]]:
    split = (bundle.get("sample") or {}).get("split") or (bundle.get("reproducibility") or {}).get("split") or {}
    if not isinstance(split, Mapping):
        return None, None, None
    train = split.get("train_user_ids")
    val = split.get("validation_user_ids")
    test = split.get("test_user_ids")
    if not (isinstance(train, list) and isinstance(val, list)):
        return None, None, None
    if not isinstance(test, list):
        test = []
    return set(map(str, train)), set(map(str, val)), set(map(str, test))


def _source_leakage_checks() -> List[Dict[str, Any]]:
    section = "DATA LEAKAGE"
    runner_path = Path(__file__).with_name("pandora_runner.py")
    source = _read_text(runner_path)
    out: List[Dict[str, Any]] = []
    if not source:
        out.append(_check(
            section, "data_leakage.runner_source_readable", "WARNING",
            f"Could not read {runner_path.name} for static leakage checks.",
        ))
        return out

    held_out_transform = (
        "transform_features(X_te" in source
        or "transform_features(X_val" in source
    )
    if "prepare_training_data(X_tr" in source and held_out_transform:
        out.append(_check(
            section, "data_leakage.lasso_scaler_train_only", "PASS",
            "Lasso feature scaler is fit on training embeddings and applied to the held-out fold via transform_features.",
        ))
    else:
        out.append(_check(
            section, "data_leakage.lasso_scaler_train_only", "FAIL",
            "Could not confirm that the Lasso scaler is fit on training data only.",
            missing="prepare_training_data(X_tr ...) then transform_features(X_val|X_te)",
        ))

    if (
        "_paired_gan_samples(X_tr" in source
        or "gan.fit(X_tr" in source
        or ".fit(X_tr, ocean_scores=" in source
        or "_augment_pooled_gan(X_tr" in source
    ):
        out.append(_check(
            section, "data_leakage.gan_fit_train_only", "PASS",
            "GAN fit/generation is sourced from training pairs only.",
        ))
    else:
        out.append(_check(
            section, "data_leakage.gan_fit_train_only", "FAIL",
            "Could not confirm GAN is fit on training participants only.",
            missing="_augment_pooled_gan(X_tr) / .fit(X_tr, ocean_scores=...)",
        ))

    run_all_src = source
    marker = "def run_all("
    start = source.find(marker)
    if start >= 0:
        nxt = source.find("\ndef ", start + len(marker))
        run_all_src = source[start:nxt if nxt > 0 else start + 4000]
    q_pos = run_all_src.find("train_qlearning_agent(")
    split_pos = run_all_src.find("make_split(")
    if q_pos >= 0 and split_pos >= 0 and q_pos < split_pos:
        if "qlearning_train_epochs <= 0" in source:
            out.append(_check(
                section, "data_leakage.qlearning_train_before_split", "FAIL",
                "Q-learning is trained on sample.texts before make_split, so test participants' comments can enter policy training.",
                missing="train Q-learning on training participants only, then freeze the policy",
                condition="train_qlearning_agent(sample) precedes make_split",
            ))
        else:
            out.append(_check(
                section, "data_leakage.qlearning_train_before_split", "FAIL",
                "Q-learning training occurs before the participant-level split.",
                missing="train Q-learning on training participants only",
            ))
    elif q_pos >= 0 and split_pos >= 0:
        out.append(_check(
            section, "data_leakage.qlearning_train_before_split", "PASS",
            "Q-learning training is ordered after the participant-level split in run_all.",
        ))
    else:
        out.append(_check(
            section, "data_leakage.qlearning_train_before_split", "WARNING",
            "Could not locate train_qlearning_agent / make_split ordering in run_all.",
        ))
    return out


def _artifact_leakage_checks(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    section = "DATA LEAKAGE"
    out: List[Dict[str, Any]] = []
    train, val, test = _split_sets(bundle)
    if train is None or val is None:
        out.append(_check(
            section, "data_leakage.participant_level_split_recorded", "FAIL",
            "Participant-level fold membership (train/validation user ids) is not recorded.",
            missing="sample.split.train_user_ids / validation_user_ids",
        ))
    else:
        two_fold = not test or test == val
        overlap_tv = train & val
        overlap_tt = train & test if test and not two_fold else set()
        overlap_vt = val & test if test and not two_fold else set()
        if overlap_tv or overlap_tt or overlap_vt:
            parts = []
            if overlap_tv:
                parts.append(f"train∩val={sorted(overlap_tv)[:5]}")
            if overlap_tt:
                parts.append(f"train∩test={sorted(overlap_tt)[:5]}")
            if overlap_vt:
                parts.append(f"val∩test={sorted(overlap_vt)[:5]}")
            out.append(_check(
                section, "data_leakage.participant_folds_disjoint", "FAIL",
                "Participant-level split is not disjoint: " + "; ".join(parts),
                missing="disjoint train/validation participant ids",
            ))
        else:
            out.append(_check(
                section, "data_leakage.participant_folds_disjoint", "PASS",
                f"Participant-level split is disjoint (train={len(train)}, val={len(val)}"
                + ("" if two_fold else f", test={len(test)}")
                + ").",
            ))
        if two_fold:
            if overlap_tv:
                out.append(_check(
                    section, "data_leakage.test_not_in_training", "FAIL",
                    "Held-out validation participants also appear in the training fold.",
                    missing="held-out participants excluded from training",
                ))
            else:
                out.append(_check(
                    section, "data_leakage.test_not_in_training", "PASS",
                    "Held-out validation participants do not appear in the training fold.",
                ))
            out.append(_check(
                section, "data_leakage.two_fold_held_out", "WARNING",
                "This runner records a train/validation split; the validation fold is the held-out eval set. A third unused test fold is not present.",
            ))
        elif overlap_tt:
            out.append(_check(
                section, "data_leakage.test_not_in_training", "FAIL",
                "Test participants also appear in the training fold.",
                missing="test participants excluded from training",
                condition=",".join(sorted(overlap_tt)[:8]),
            ))
        else:
            out.append(_check(
                section, "data_leakage.test_not_in_training", "PASS",
                "Recorded test participants do not appear in the training fold.",
            ))

    scale = (
        (bundle.get("sample") or {}).get("label_scale")
        or ((bundle.get("reproducibility") or {}).get("preprocessing") or {}).get("label_scale")
        or ""
    )
    scale_text = str(scale).lower()
    if "min-max" in scale_text:
        out.append(_check(
            section, "data_leakage.label_scale_train_only", "FAIL",
            f"Label min-max scaling used the full sample range ({scale}), which can include test labels.",
            missing="fit min-max on training labels only",
        ))
    elif scale_text:
        out.append(_check(
            section, "data_leakage.label_scale_train_only", "WARNING",
            f"Label scale '{scale}' was detected on the sampled matrix before the split. "
            "Fixed affine maps (unit/Likert/percentile) do not estimate test statistics; "
            "min-max fallback would.",
        ))
    else:
        out.append(_check(
            section, "data_leakage.label_scale_train_only", "WARNING",
            "Label scale was not recorded; cannot confirm transforms avoided test information.",
            missing="sample.label_scale",
        ))

    cfg = bundle.get("config") or {}
    epochs = cfg.get("qlearning_train_epochs")
    if _as_int(epochs) == 0:
        out.append(_check(
            section, "data_leakage.qlearning_untrained_policy", "WARNING",
            "Q-learning epochs are 0 (cold policy), so test comments did not update the Q-table, "
            "but Q-learning cells are not a trained selection policy.",
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _overall_status(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {item.get("status") for item in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS"


def _validate_single(bundle: Mapping[str, Any], *, include_source: bool = True) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    checks.extend(_reproducibility_checks(bundle))
    checks.extend(_data_quality_checks(bundle))
    checks.extend(_prediction_quality_checks(bundle))
    checks.extend(_interpretation_checks(bundle))
    checks.extend(_experiment_design_checks(bundle))
    checks.extend(_artifact_leakage_checks(bundle))
    if include_source:
        checks.extend(_source_leakage_checks())
    return checks


def validate_research_contract(
    source: Optional[Payload] = None,
    *,
    include_source_checks: bool = True,
) -> Dict[str, Any]:
    """
    Validate a run bundle, saved JSON, output directory, or source-only contract.

    ``source=None`` still runs static leakage/source checks and reports missing
    artifacts as FAIL. That is the pre-run readiness view.
    """
    checks: List[Dict[str, Any]] = []
    scope = "source_only"
    seeds_validated: List[Any] = []

    if source is None:
        checks.extend(_source_leakage_checks() if include_source_checks else [])
        checks.append(_check(
            "REPRODUCIBILITY", "reproducibility.artifacts_present", "FAIL",
            "No run payload was supplied; recorded seed/SHA/config/folds cannot be verified.",
            missing="run bundle or output directory",
        ))
        checks.append(_check(
            "DATA QUALITY", "data_quality.artifacts_present", "FAIL",
            "No run payload was supplied; data-quality counts cannot be verified.",
            missing="data_quality",
        ))
        checks.append(_check(
            "PREDICTION QUALITY", "prediction_quality.artifacts_present", "FAIL",
            "No run payload was supplied; prediction metrics cannot be verified.",
            missing="results",
        ))
        checks.append(_check(
            "INTERPRETATION", "interpretation.artifacts_present", "FAIL",
            "No run payload was supplied; interpretation cannot be verified.",
            missing="interpretation",
        ))
        checks.append(_check(
            "EXPERIMENT DESIGN", "experiment_design.artifacts_present", "FAIL",
            "No run payload was supplied; the 8-condition matrix cannot be verified.",
            missing="results",
        ))
    else:
        bundle = load_run_payload(source)
        if bundle.get("kind") == "multi_seed_experiment":
            scope = "multi_seed"
            runs = bundle.get("runs") or {}
            source_once = include_source_checks
            if not runs:
                checks.append(_check(
                    "EXPERIMENT DESIGN", "experiment_design.multi_seed_runs", "FAIL",
                    "Multi-seed payload has no per-seed runs.",
                    missing="runs",
                ))
            for seed, run in runs.items():
                if not isinstance(run, Mapping):
                    checks.append(_check(
                        "EXPERIMENT DESIGN", "experiment_design.seed_run", "FAIL",
                        f"Seed {seed} run is not an object.",
                        missing=f"runs[{seed}]",
                        condition=str(seed),
                    ))
                    continue
                seeds_validated.append(seed)
                prefixed = _validate_single(run, include_source=source_once)
                source_once = False
                for item in prefixed:
                    item = dict(item)
                    item["seed"] = seed
                    checks.append(item)
            if not (bundle.get("aggregation") or {}).get("conditions"):
                checks.append(_check(
                    "PREDICTION QUALITY", "prediction_quality.multi_seed_aggregation", "FAIL",
                    "Multi-seed aggregation of MAE/RMSE/R²/F1/ROC/PR is missing.",
                    missing="aggregation.conditions",
                ))
            else:
                checks.append(_check(
                    "PREDICTION QUALITY", "prediction_quality.multi_seed_aggregation", "PASS",
                    "Multi-seed metric aggregation is present.",
                ))
            if not bundle.get("paired_differences"):
                checks.append(_check(
                    "EXPERIMENT DESIGN", "experiment_design.multi_seed_paired_differences", "FAIL",
                    "Same-seed paired differences are missing.",
                    missing="paired_differences",
                ))
            else:
                checks.append(_check(
                    "EXPERIMENT DESIGN", "experiment_design.multi_seed_paired_differences", "PASS",
                    "Same-seed paired differences are present.",
                ))
        else:
            scope = "single_run"
            checks.extend(_validate_single(bundle, include_source=include_source_checks))

    status = _overall_status(checks)
    fails = [item for item in checks if item["status"] == "FAIL"]
    warnings = [item for item in checks if item["status"] == "WARNING"]
    return {
        "kind": "research_contract_report",
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "seeds_validated": seeds_validated,
        "counts": {
            "pass": sum(1 for item in checks if item["status"] == "PASS"),
            "warning": len(warnings),
            "fail": len(fails),
            "total": len(checks),
        },
        "fails": fails,
        "warnings": warnings,
        "checks": checks,
        "stops_execution": False if status != "FAIL" else False,
        "warning_stops_execution": False,
        "note": (
            "WARNING does not stop execution. FAIL names the exact missing or "
            "invalid condition and is a research-readiness gate, not a training abort."
        ),
    }


def format_report(report: Mapping[str, Any]) -> str:
    lines = [
        "=" * 72,
        f"RESEARCH CONTRACT  overall: {report.get('status')}",
        f"scope={report.get('scope')}  "
        f"PASS={report.get('counts', {}).get('pass', 0)}  "
        f"WARNING={report.get('counts', {}).get('warning', 0)}  "
        f"FAIL={report.get('counts', {}).get('fail', 0)}",
        "=" * 72,
    ]
    current = None
    for item in report.get("checks") or []:
        if item.get("section") != current:
            current = item.get("section")
            lines.append("")
            lines.append(current)
        loc = ""
        if item.get("condition"):
            loc += f" [{item['condition']}]"
        if item.get("seed") is not None:
            loc += f" seed={item['seed']}"
        lines.append(f"  {item['status']:<7} {item['id']}{loc}")
        lines.append(f"          {item['detail']}")
        if item.get("missing"):
            lines.append(f"          missing/invalid: {item['missing']}")
    if report.get("fails"):
        lines.append("")
        lines.append("FAILS (exact missing or invalid conditions)")
        for item in report["fails"]:
            loc = item.get("condition") or item.get("id")
            lines.append(f"  - {loc}: {item.get('missing') or item.get('detail')}")
    lines.append("")
    lines.append(str(report.get("note") or ""))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the PANDORA research contract (readiness gate).",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Run output directory, run_summary.json, or multi_seed_summary.json",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Write the machine-readable report to this path",
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        help="Skip static runner-source leakage checks",
    )
    args = parser.parse_args(argv)
    report = validate_research_contract(
        args.path,
        include_source_checks=not args.skip_source,
    )
    text = format_report(report)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    if report.get("status") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
