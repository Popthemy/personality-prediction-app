"""Django-free experiment engine for the ML pipeline.

This package exposes the PANDORA experiment runner without importing the heavy
ML stack until one of the public objects is actually requested. That keeps
``python -m backend.ml_pipeline.experiments.pandora_runner`` quiet and avoids
surprising imports for tooling that only inspects the package.
"""

_PANDORA_EXPORTS = {
    "ExperimentConfig",
    "ExperimentRunner",
    "EXPERIMENTS",
    "OCEAN_TRAITS",
    "TRAIT_KEYS",
    "prepare_sample",
    "load_prepared_cache",
    "load_or_prepare_pandora",
    "run_condition",
    "run_all",
    "comparison_table",
    "factor_effects",
    "summarize_findings",
}

__all__ = sorted(_PANDORA_EXPORTS)


def __getattr__(name):
    if name in _PANDORA_EXPORTS:
        from . import pandora_runner

        return getattr(pandora_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
