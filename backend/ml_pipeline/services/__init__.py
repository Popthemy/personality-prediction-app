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
from .pipeline_orchestrator import PipelineOrchestrator

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
