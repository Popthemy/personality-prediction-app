"""Django-free experiment engine for the ML pipeline.

This package reproduces the E1-E4 experiment glue that normally lives in the
Django-coupled ``pipeline_orchestrator`` (which reads/writes the ORM), but drives
the *same* Django-free service classes directly on in-memory PANDORA data. It is
what the Google Colab notebook imports and calls.

Nothing here imports Django. See ``pandora_runner`` for the public API.
"""

from .pandora_runner import (
    ExperimentConfig,
    ExperimentRunner,
    EXPERIMENTS,
    OCEAN_TRAITS,
    prepare_sample,
    run_experiment,
    run_all,
    comparison_table,
    model_comparison,
    factor_effects,
    hybrid_cell_evaluations,
    summarize_findings,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "EXPERIMENTS",
    "OCEAN_TRAITS",
    "prepare_sample",
    "run_experiment",
    "run_all",
    "comparison_table",
    "model_comparison",
    "factor_effects",
    "hybrid_cell_evaluations",
    "summarize_findings",
]
