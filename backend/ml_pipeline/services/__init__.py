# """ML Pipeline services."""
# from .qlearning_agent import QLearningAgent, create_post_features
# from .bert_encoder import BERTEncoder, create_bert_embeddings_for_posts
# from .gan_augmenter import GANAugmenter
# from .lasso_regressor import LassoTrainer, denormalize_predictions
# from .pipeline_orchestrator import PipelineOrchestrator

# __all__ = [
#     'QLearningAgent',
#     'BERTEncoder',
#     'GANAugmenter',
#     'LassoTrainer',
#     'PipelineOrchestrator',
#     'create_post_features',
#     'create_bert_embeddings_for_posts',
#     'denormalize_predictions',
# ]



"""ML Pipeline services."""
from .qlearning_agent import (
    QLearningAgent,
    # create_post_features,
    # New public additions (do not remove; pipeline_orchestrator may use these)
    CommentSelectionEnvironment,
    PostSelectionEnvironment,   # backward-compat alias → CommentSelectionEnvironment
    run_training_episode,
    run_training_loop,
    EpisodeStats,
)
from .bert_encoder import BERTEncoder, create_bert_embeddings_for_posts
from .gan_augmenter import GANAugmenter
from .lasso_regressor import LassoTrainer, denormalize_predictions

# NOTE: PipelineOrchestrator is intentionally NOT imported eagerly here.
# It depends on Django (django.db + backend.core.models), so importing it at
# package-import time forces every consumer of these leaf services to have
# Django configured. The Colab / experiment path
# (backend.ml_pipeline.experiments.pandora_runner) reuses the Django-free
# services above WITHOUT Django installed, and it imports modules under this
# package — which runs this __init__. Orchestrator is therefore exposed lazily
# via __getattr__ (PEP 562) below, so
# `from backend.ml_pipeline.services import PipelineOrchestrator` still works
# unchanged inside the Django app, but merely importing e.g. QLearningAgent or
# the PANDORA loader no longer drags in Django.

__all__ = [
    # Core agents
    'QLearningAgent',
    'BERTEncoder',
    'GANAugmenter',
    'LassoTrainer',
    'PipelineOrchestrator',
    # Q-learning environment & training helpers
    'CommentSelectionEnvironment',
    'PostSelectionEnvironment',
    'run_training_episode',
    'run_training_loop',
    'EpisodeStats',
    # Helper functions
    # 'create_post_features',
    'create_bert_embeddings_for_posts',
    'denormalize_predictions',
]


def __getattr__(name):
    """Lazily expose Django-coupled exports (PEP 562).

    Keeps ``from backend.ml_pipeline.services import PipelineOrchestrator``
    working for the Django app, while allowing the Django-free services above to
    be imported (e.g. on Google Colab) without Django installed.
    """
    if name == "PipelineOrchestrator":
        from .pipeline_orchestrator import PipelineOrchestrator

        return PipelineOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
