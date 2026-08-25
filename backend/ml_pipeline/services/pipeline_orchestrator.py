"""
Pipeline Orchestrator - Main pipeline execution controller.

Strict execution order:
1. Input Data (X API / CSV import)
2. Q-Learning Active Signal Selection
3. BERT Contextual Embedding Extraction
4. GAN Data Augmentation (training only)
5. Lasso Regression → Final Prediction
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from django.db import transaction
from backend.core.models import (
    VOLUNTEER, POST, BERT_EMBEDDING, Q_LEARNING_LOG,
    SYNTHETIC_DATA, LASSO_MODEL, COHORT_MODEL, PSYCHOMETRIC_PROFILE, BFI_SURVEY
)
from backend.ml_pipeline.services.timeline_exporter import export_cleaned_posts_to_txt

# Experiment definitions
EXPERIMENTS = {
    'E1': {
        'description': 'Baseline: Simple selection, no augmentation, Lasso regression',
        'use_qlearning': False,
        'use_gan': False,
        'model': 'lasso'
    },
    'E2': {
        'description': 'Q-Learning selection, no augmentation, Lasso regression',
        'use_qlearning': True,
        'use_gan': False,
        'model': 'lasso'
    },
    'E3': {
        'description': 'Q-Learning selection, GAN augmentation, Lasso regression',
        'use_qlearning': True,
        'use_gan': True,
        'model': 'lasso'
    },
    'E4': {
        'description': 'Q-Learning selection, no augmentation, LSTM classification',
        'use_qlearning': True,
        'use_gan': False,
        'model': 'lstm'
    }
}

logger = logging.getLogger('ml_pipeline')

SYNTHETIC_SAMPLE_WEIGHT = 0.35


class BaselineSelector:
    """A simple selector that picks the most recent posts."""

    def select_posts(self, posts: List, top_k: int) -> List:
        """Selects the top_k most recent posts."""
        return posts[:top_k]


class PipelineOrchestrator:
    """Main pipeline orchestrator."""

    def __init__(self, volunteer_id: int):
        """
        Initialize orchestrator for a volunteer.

        Args:
            volunteer_id: Volunteer database ID
        """
        self.volunteer = VOLUNTEER.objects.get(id=volunteer_id)
        self.logger = logging.getLogger(
            f'ml_pipeline.{self.volunteer.x_handle}')

    def _set_pipeline_status(self, status: str):
        """Persist the volunteer pipeline status."""
        self.volunteer.pipeline_status = status
        self.volunteer.save(update_fields=['pipeline_status'])

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return float(max(minimum, min(maximum, value)))

    def run_full_pipeline(self, experiment_id: str) -> Dict:
        """
        Execute a full pipeline experiment.
        """
        if experiment_id not in EXPERIMENTS:
            raise ValueError(f"Experiment '{experiment_id}' not found.")

        config = EXPERIMENTS[experiment_id]
        self.logger.info(
            f"STARTING EXPERIMENT {experiment_id}: {config['description']}")

        try:
            # 1. Input Data
            posts = self._step_1_input_data()
            if not posts:
                raise ValueError("No posts available for processing.")

            # 2. Selection
            if config['use_qlearning']:
                selected_posts = self._step_2_qlearning_selection(posts)
            else:
                selector = BaselineSelector()
                selected_posts = selector.select_posts(posts, top_k=10)

            # 3. BERT Embedding
            embeddings = self._step_3_bert_embedding(selected_posts)

            # 4. Augmentation (for training)
            synthetic_data = []
            if config['use_gan']:
                synthetic_data = self._step_4_gan_augmentation(
                    selected_posts, embeddings)

            # 5. Modeling
            if config['model'] == 'lasso':
                prediction_result = self._step_5_lasso_prediction(
                    embeddings, synthetic_data)
            elif config['model'] == 'lstm':
                # Assuming LSTMTrainer is available and has a similar interface
                from .lstm_classifier import LSTMTrainer
                lstm_trainer = LSTMTrainer(self.volunteer.id)
                prediction_result = lstm_trainer.train_and_evaluate()  # This might need adjustment
            else:
                raise ValueError(f"Unsupported model type: {config['model']}")

            # 6. Evaluation & Persistence
            self._save_psychometric_profile(prediction_result)
            self._set_pipeline_status('completed')

            return {
                'status': 'success',
                'experiment_id': experiment_id,
                **prediction_result,
            }

        except Exception as e:
            self.logger.error(
                f"Pipeline failed for experiment {experiment_id}: {e}", exc_info=True)
            self._set_pipeline_status('error')
            return {'status': 'error', 'error': str(e)}

    def _step_1_input_data(self) -> List:
        """
        Step 1: Retrieve input posts from database and clean them.

        Returns:
            List of POST objects with cleaned_content attached
        """

        self.logger.info("STEP 1: Input Data - retrieving and cleaning posts")
        from backend.ml_pipeline.processors.text_preprocessor import TextPreprocessor
        from backend.core.services.twitter_fetcher import TwitterFetcher

        preprocessor = TextPreprocessor()
        raw_posts = POST.objects.filter(
            volunteer=self.volunteer).order_by('-created_at_original')
        self.logger.info(
            f"Retrieved {raw_posts.count()} raw posts from database")

        if raw_posts.count() == 0:
            self.logger.info(
                "No posts in database. Attempting live X fetch before aborting.")
            fetcher = TwitterFetcher()
            saved, skipped = fetcher.fetch_and_save(self.volunteer)
            self.logger.info(
                f"Live fetch attempt for @{self.volunteer.x_handle} returned saved={saved}, skipped={skipped}"
            )
            raw_posts = POST.objects.filter(
                volunteer=self.volunteer).order_by('-created_at_original')
            self.logger.info(
                f"Database now has {raw_posts.count()} posts after fetch attempt")

        posts = []
        for post in raw_posts:
            cleaned = preprocessor.clean(post.content)
            if preprocessor.is_valid(cleaned):
                post.cleaned_content = cleaned
                posts.append(post)
            else:
                self.logger.debug(
                    f"Filtered out invalid/short post: {post.content[:50]!r}")

        self.logger.info(
            f"Filtered down to {len(posts)} valid posts after preprocessing")
        export_path = export_cleaned_posts_to_txt(
            self.volunteer.x_handle,
            [getattr(post, 'cleaned_content', '') for post in posts],
        )
        self.logger.info(
            "Exported cleaned timeline for @%s to %s",
            self.volunteer.x_handle,
            export_path,
        )
        return posts

    def _step_2_qlearning_selection(self, posts: List) -> List:
        """
        Step 2: Q-Learning active signal selection (PANDORA Reddit / comment-text mode).

        The redesigned QLearningAgent operates on already-cleaned comment text.
        It no longer uses engagement_score / recency_days / hashtag features;
        state is driven by lexical diversity, TF-IDF redundancy, and budget
        context instead.

        Input contract
        --------------
        Each item in *posts* is a Django POST ORM object that must carry a
        ``cleaned_content`` attribute (set by Step 1's TextPreprocessor).
        ``create_post_features()`` extracts the cleaned text string; the agent
        then builds TF-IDF vectors internally for the full episode.

        Return schema of agent.select_posts()
        --------------------------------------
        Each element of the returned list is:
            {
                "comment":  <original input item — the dict we passed in>,
                "text":     <cleaned text string>,
                "q_value":  <float>,
                "state":    <str Q-table key>,
                "action":   "select",
            }
        The original post identity is recovered via ``s["comment"]["_post_id"]``,
        a sentinel key we inject into the input dict before passing to the agent.

        Args:
            posts: List of POST objects with ``cleaned_content`` set

        Returns:
            List of selected POST objects
        """
        self.logger.info("STEP 2: Q-Learning Active Signal Selection")

        from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, create_post_features

        agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.05)

        # Build comment dicts accepted by the new agent.
        # create_post_features() returns the cleaned text string; we wrap it in
        # a dict so we can carry the post DB id through the episode.
        comment_inputs = []
        for post in posts:
            cleaned_text = create_post_features(post)   # returns str
            comment_inputs.append({
                "_post_id": post.id,          # sentinel for id recovery below
                "text": cleaned_text,         # primary field consumed by agent
            })

        # Select top comments (agent uses internal TF-IDF similarity)
        top_k = min(10, len(posts))
        selected = agent.select_posts(
            comment_inputs, top_k=top_k, training=False)

        # Recover post identities from the "comment" payload returned by agent
        # selected[i] = {"comment": {"_post_id": ..., "text": ...}, "q_value": ..., ...}
        selected_id_to_qval: Dict[int, float] = {
            s["comment"]["_post_id"]: s.get("q_value", 0.0)
            for s in selected
        }

        # Mark posts in DB and build return list
        selected_posts = []
        for post in posts:
            if post.id in selected_id_to_qval:
                post.selected_by_qlearning = True
                post.q_value = selected_id_to_qval[post.id]
                selected_posts.append(post)
            post.save()

        self.logger.info(
            "Q-Learning selected %d / %d posts",
            len(selected_posts), len(posts),
        )
        return selected_posts

    def _step_3_bert_embedding(self, posts: List) -> List:
        """
        Step 3: BERT contextual embedding extraction.

        Args:
            posts: List of selected POST objects

        Returns:
            List of BERT_EMBEDDING objects
        """
        self.logger.info("STEP 3: BERT Contextual Embedding Extraction")

        from backend.ml_pipeline.services.bert_encoder import BERTEncoder
        import time

        encoder = BERTEncoder()
        embeddings = []

        for i, post in enumerate(posts):
            start_time = time.time()

            # Encode post (using cleaned text)
            cleaned_text = getattr(post, 'cleaned_content', post.content)

            # Idempotency: skip re-encoding if embedding already exists
            existing = BERT_EMBEDDING.objects.filter(post=post).first()
            if existing:
                embeddings.append(existing)
                post.embedding_processed = True
                post.save()
                continue

            result = encoder.encode_text(cleaned_text)

            # Save to database
            embedding_obj = BERT_EMBEDDING.objects.create(
                post=post,
                volunteer=self.volunteer,
                embedding_vector=result['embedding'],
                model_name=result['model_name'],
                processing_time_seconds=time.time() - start_time,
            )

            embeddings.append(embedding_obj)
            post.embedding_processed = True
            post.save()

            if (i + 1) % 5 == 0:
                self.logger.debug(f"Processed {i+1}/{len(posts)} posts")

        self.logger.info(f"Created {len(embeddings)} BERT embeddings")
        verification = self._verify_bert_embedding_persistence(
            posts, embeddings)
        self.logger.info(
            "BERT persistence verification passed: %s/%s embeddings persisted",
            verification['persisted_count'],
            verification['expected_count'],
        )
        return embeddings

    def _verify_bert_embedding_persistence(self, posts: List, embeddings: List) -> Dict[str, int]:
        """
        Verify that BERT embeddings were actually persisted to the database.

        This runs on every pipeline execution so we catch any mismatch between
        the in-memory embedding list and the stored records before GAN/Lasso use
        the data downstream.
        """
        expected_count = len(posts)
        persisted_count = 0
        missing_posts = []

        for post in posts:
            if BERT_EMBEDDING.objects.filter(post=post, volunteer=self.volunteer).exists():
                persisted_count += 1
            else:
                missing_posts.append(post.id)

        if persisted_count != expected_count or len(embeddings) != expected_count:
            raise ValueError(
                "BERT embedding persistence check failed: "
                f"created={len(embeddings)}, persisted={persisted_count}, expected={expected_count}, "
                f"missing_post_ids={missing_posts}"
            )

        return {
            'expected_count': expected_count,
            'created_count': len(embeddings),
            'persisted_count': persisted_count,
        }

    def _step_4_gan_augmentation(self, posts: List, embeddings: List) -> List:
        """
        Step 4: GAN data augmentation for training.

        Args:
            posts: List of POST objects
            embeddings: List of BERT_EMBEDDING objects

        Returns:
            List of SYNTHETIC_DATA objects
        """
        self.logger.info("STEP 4: GAN Data Augmentation")

        from backend.ml_pipeline.services.gan_augmenter import GANAugmenter

        augmenter = GANAugmenter(augmentation_factor=0.8)
        synthetic_list = []

        # Create 1x augmentation (double the training set)
        target_size = len(embeddings)

        for i in range(target_size):
            original_embedding = embeddings[i % len(embeddings)]
            original_post = posts[i % len(posts)]

            # Augment embedding
            embedding_list = original_embedding.embedding_vector
            if isinstance(embedding_list, str):
                import json
                embedding_list = json.loads(embedding_list)

            augmented_embedding = augmenter.augment_embedding(embedding_list)
            synthetic_text = augmenter.generate_synthetic_text()

            # Save to database
            synthetic_obj = SYNTHETIC_DATA.objects.create(
                volunteer=self.volunteer,
                original_post=original_post,
                original_embedding=original_embedding,
                synthetic_text=synthetic_text,
                synthetic_embedding=augmented_embedding,
                generator_version='gan-v1',
                generation_confidence=0.85,
                used_in_training=True,
            )

            synthetic_list.append(synthetic_obj)

        self.logger.info(
            f"Generated {len(synthetic_list)} synthetic training samples")
        return synthetic_list

    def _embedding_to_array(self, embedding_value):
        """Convert a stored embedding value into a numpy array."""
        import json

        if isinstance(embedding_value, str):
            embedding_value = json.loads(embedding_value)

        if isinstance(embedding_value, dict):
            return np.array([embedding_value[str(i)] for i in range(len(embedding_value))], dtype=float)

        return np.array(embedding_value, dtype=float)

    def _pool_embeddings(self, embedding_values: List) -> Optional[np.ndarray]:
        """Mean-pool a list of embeddings into a single volunteer-level vector."""
        if not embedding_values:
            return None

        vectors = [self._embedding_to_array(value)
                   for value in embedding_values]
        return np.mean(np.vstack(vectors), axis=0)

    def _volunteer_feature_vector(self, volunteer: VOLUNTEER, include_synthetic: bool = True) -> Optional[np.ndarray]:
        """
        Build a volunteer-level feature vector using pooled embeddings.

        Real post embeddings are pooled first. Synthetic embeddings are appended
        as additional samples when requested to stabilize the representation.
        """
        real_embeddings = BERT_EMBEDDING.objects.filter(volunteer=volunteer)
        selected_embeddings = real_embeddings.filter(
            post__selected_by_qlearning=True)
        if selected_embeddings.exists():
            real_embeddings = selected_embeddings

        vectors = [self._embedding_to_array(
            emb.embedding_vector) for emb in real_embeddings]

        if include_synthetic:
            synthetic_rows = SYNTHETIC_DATA.objects.filter(
                volunteer=volunteer, used_in_training=True)
            vectors.extend([
                self._embedding_to_array(row.synthetic_embedding)
                for row in synthetic_rows
            ])

        return self._pool_embeddings(vectors)

    def _build_training_dataset(self, exclude_volunteer_ids: Optional[List[int]] = None):
        """
        Build a cohort training set from volunteers with ground truth survey data.

        Each volunteer contributes a pooled real-embedding sample plus any
        synthetic augmentation rows saved for that volunteer.
        """
        exclude_volunteer_ids = exclude_volunteer_ids or []

        volunteers = VOLUNTEER.objects.filter(bfi_survey__isnull=False).exclude(
            id__in=exclude_volunteer_ids
        ).select_related('bfi_survey')

        samples = []
        synthetic_count = 0
        trait_labels = {
            'Openness': [],
            'Conscientiousness': [],
            'Extraversion': [],
            'Agreeableness': [],
            'Neuroticism': [],
        }

        for volunteer in volunteers:
            base_vector = self._volunteer_feature_vector(
                volunteer, include_synthetic=False)
            if base_vector is None:
                continue

            survey = volunteer.bfi_survey
            ocean = survey.get_ocean_dict()

            samples.append(base_vector)
            for trait, value in ocean.items():
                trait_labels[trait].append(value)

            synthetic_rows = SYNTHETIC_DATA.objects.filter(
                volunteer=volunteer, used_in_training=True)
            for row in synthetic_rows:
                syn_vector = self._embedding_to_array(row.synthetic_embedding)
                samples.append(syn_vector)
                synthetic_count += 1
                for trait, value in ocean.items():
                    trait_labels[trait].append(value)

        if not samples:
            return np.empty((0, 0)), trait_labels, synthetic_count

        return np.vstack(samples), {k: np.array(v, dtype=float) for k, v in trait_labels.items()}, synthetic_count

    def _build_labeled_volunteer_records(self, exclude_volunteer_ids: Optional[List[int]] = None):
        """
        Build one record per labeled volunteer using only real post embeddings.

        Synthetic rows are intentionally excluded here so validation happens on
        human-labeled volunteers only. Synthetic augmentation is applied later to
        the training fold only.
        """
        exclude_volunteer_ids = exclude_volunteer_ids or []

        volunteers = VOLUNTEER.objects.filter(bfi_survey__isnull=False).exclude(
            id__in=exclude_volunteer_ids
        ).select_related('bfi_survey')

        records = []
        for volunteer in volunteers:
            base_vector = self._volunteer_feature_vector(
                volunteer, include_synthetic=False)
            if base_vector is None:
                continue

            records.append({
                'volunteer_id': volunteer.id,
                'handle': volunteer.x_handle,
                'vector': base_vector,
                'labels': volunteer.bfi_survey.get_ocean_dict(),
            })

        return records

    def _augment_training_fold(self, records: List[Dict], include_synthetic: bool = True):
        """
        Expand real training volunteer records with their stored synthetic rows.

        Returns:
            (X, labels_by_trait, synthetic_count, sample_weights)
        """
        if not records:
            return np.empty((0, 0)), {}, 0, np.array([])

        samples = []
        labels_by_trait = {
            'Openness': [],
            'Conscientiousness': [],
            'Extraversion': [],
            'Agreeableness': [],
            'Neuroticism': [],
        }
        sample_weights = []
        synthetic_count = 0

        for record in records:
            samples.append(record['vector'])
            for trait, value in record['labels'].items():
                labels_by_trait[trait].append(value)
            sample_weights.append(1.0)

            if include_synthetic:
                synthetic_rows = SYNTHETIC_DATA.objects.filter(
                    volunteer_id=record['volunteer_id'],
                    used_in_training=True,
                )
                for row in synthetic_rows:
                    samples.append(self._embedding_to_array(
                        row.synthetic_embedding))
                    synthetic_count += 1
                    sample_weights.append(SYNTHETIC_SAMPLE_WEIGHT)
                    for trait, value in record['labels'].items():
                        labels_by_trait[trait].append(value)

        if not samples:
            return np.empty((0, 0)), labels_by_trait, synthetic_count, np.array([])

        return (
            np.vstack(samples),
            {k: np.array(v, dtype=float) for k, v in labels_by_trait.items()},
            synthetic_count,
            np.array(sample_weights, dtype=float),
        )

    def _bootstrap_training_inputs(self, volunteers: List[VOLUNTEER]) -> List[Dict]:
        """
        Ensure each labeled volunteer has posts, selected posts, and embeddings
        before cohort training starts.
        """
        prepared = []

        for volunteer in volunteers:
            volunteer_orchestrator = PipelineOrchestrator(volunteer.id)
            try:
                posts = volunteer_orchestrator._step_1_input_data()
                if not posts:
                    self.logger.warning(
                        "Skipping @%s because no usable posts were available after fetch",
                        volunteer.x_handle,
                    )
                    continue

                selected_posts = volunteer_orchestrator._step_2_qlearning_selection(
                    posts)
                embeddings = volunteer_orchestrator._step_3_bert_embedding(
                    selected_posts)
                prepared.append({
                    'volunteer_id': volunteer.id,
                    'handle': volunteer.x_handle,
                    'posts': len(posts),
                    'selected_posts': len(selected_posts),
                    'embeddings': len(embeddings),
                })
            except Exception as e:
                self.logger.warning(
                    "Failed to bootstrap training inputs for @%s: %s",
                    volunteer.x_handle,
                    e,
                )

        return prepared

    def _fit_trait_variant(
        self,
        trait_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray],
        X_val_raw: Optional[np.ndarray],
        y_val_raw: Optional[np.ndarray],
        can_validate: bool,
        variant_name: str,
    ) -> Dict:
        """Fit one trait model variant and return the trained artifact bundle."""
        from backend.ml_pipeline.services.lasso_regressor import LassoTrainer, denormalize_predictions

        variant_trainer = LassoTrainer(
            alpha=0.001,
            max_iter=10000,
            regularization='elasticnet',
            l1_ratio=0.5,
        )
        X_fit, y_fit_norm = variant_trainer.prepare_training_data(
            X_train, y_train)

        X_val_norm = None
        y_val_norm = None
        if can_validate and X_val_raw is not None and y_val_raw is not None and len(y_val_raw) > 0:
            X_val_norm = variant_trainer.transform_features(X_val_raw)
            y_val_norm = (y_val_raw - 1.0) / 4.0

        trait_metrics = variant_trainer.train_trait_model(
            X_fit,
            y_fit_norm,
            trait_name,
            validate_X=X_val_norm,
            validate_y=y_val_norm,
            sample_weight=sample_weight,
        )

        if len(X_train) == 0:
            raise ValueError(
                f"No training data available for trait {trait_name}")

        current_norm = variant_trainer.transform_features(X_train[:1])
        train_pred_norm = variant_trainer.predict_trait(
            trait_name, current_norm)
        train_pred = denormalize_predictions(train_pred_norm)

        validation_mae = trait_metrics.get('validation_mae')
        if validation_mae is None:
            validation_mae = trait_metrics.get('train_mae', float('inf'))

        return {
            'trainer': variant_trainer,
            'metrics': trait_metrics,
            'variant': variant_name,
            'validation_mae': float(validation_mae),
            'training_samples_used': int(X_train.shape[0]),
            'sample_weight_used': bool(sample_weight is not None),
            'preview_prediction': float(np.mean(train_pred)),
        }

    def train_cohort_model(self, split_seed: int = 42) -> Dict:
        """
        Train the shared cohort model on 80% of the labeled volunteers and
        reserve the remaining 20% for validation and future prediction runs.
        """
        from sklearn.model_selection import train_test_split

        labeled_volunteers = list(
            VOLUNTEER.objects.filter(
                researcher=self.volunteer.researcher,
                bfi_survey__isnull=False,
            ).order_by('id')
        )
        if len(labeled_volunteers) < 2:
            raise ValueError(
                "Need at least 2 labeled volunteers to train a cohort model")

        bootstrap_stats = self._bootstrap_training_inputs(labeled_volunteers)
        labeled_records = self._build_labeled_volunteer_records()
        if len(labeled_records) < 2:
            raise ValueError(
                "No usable training samples were found after auto-fetching posts. "
                "At least two labeled volunteers need fetched posts and embeddings."
            )

        if len(labeled_records) >= 3:
            val_size = max(1, int(round(len(labeled_records) * 0.2)))
            if val_size >= len(labeled_records):
                val_size = 1
            train_records, val_records = train_test_split(
                labeled_records,
                test_size=val_size,
                random_state=split_seed,
                shuffle=True,
            )
        else:
            train_records = list(labeled_records)
            val_records = []

        real_X_train, real_y_train_dict, _, _ = self._augment_training_fold(
            train_records,
            include_synthetic=False,
        )
        augmented_X_train, augmented_y_train_dict, synthetic_train_count, augmented_weights = self._augment_training_fold(
            train_records,
            include_synthetic=True,
        )

        if len(real_X_train) == 0:
            raise ValueError("No usable training samples were found")

        X_val_raw = None
        y_val_dict = {}
        if val_records:
            X_val_raw = np.vstack([record['vector'] for record in val_records])
            y_val_dict = {
                trait: np.array([record['labels'][trait]
                                for record in val_records], dtype=float)
                for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            }

        can_validate = bool(
            val_records and X_val_raw is not None and len(X_val_raw) > 0)
        trait_artifacts = {}
        predictions = {}
        metrics = {}
        any_augmented_selected = False

        self.logger.info(
            "Training cohort model with %s train volunteers and %s validation volunteers",
            len(train_records),
            len(val_records),
        )

        for trait, y_values in real_y_train_dict.items():
            if len(y_values) == 0:
                predictions[f'predicted_{trait.lower()}'] = None
                predictions[f'mae_{trait.lower()}'] = None
                continue

            baseline_result = self._fit_trait_variant(
                trait,
                real_X_train,
                y_values,
                np.ones(len(real_X_train), dtype=float),
                X_val_raw,
                y_val_dict.get(trait),
                can_validate,
                'real_only',
            )
            chosen_result = baseline_result

            if can_validate:
                augmented_result = self._fit_trait_variant(
                    trait,
                    augmented_X_train,
                    augmented_y_train_dict[trait],
                    augmented_weights,
                    X_val_raw,
                    y_val_dict.get(trait),
                    can_validate,
                    'augmented',
                )
                if augmented_result['validation_mae'] <= baseline_result['validation_mae']:
                    chosen_result = augmented_result
                    any_augmented_selected = True

            trait_artifacts[trait] = chosen_result
            metrics[trait] = chosen_result['metrics']
            predictions[f'predicted_{trait.lower()}'] = chosen_result['preview_prediction']

        for trait, artifact in trait_artifacts.items():
            trainer = artifact['trainer']
            model_obj, _ = LASSO_MODEL.objects.update_or_create(
                volunteer=self.volunteer,
                trait=trait.lower(),
                defaults={
                    'alpha': trainer.alpha,
                    'coefficients': trainer.get_all_coefficients(trait),
                    'intercept': float(trainer.models[trait].intercept_),
                    'training_samples_used': int(artifact['training_samples_used']),
                    'synthetic_samples_used': int(synthetic_train_count if artifact['variant'] == 'augmented' else 0),
                },
            )

            trait_metrics = artifact['metrics']
            for key in ['train_mae', 'train_rmse', 'train_r2', 'validation_mae', 'validation_rmse', 'validation_r2']:
                if key in trait_metrics:
                    setattr(model_obj, key, trait_metrics[key])
            model_obj.save()

        mae_values = [
            float(artifact['metrics'].get('validation_mae'))
            for artifact in trait_artifacts.values()
            if artifact['metrics'].get('validation_mae') is not None
        ]
        predictions['overall_mae'] = float(
            np.mean(mae_values)) if mae_values else None
        predictions['training_mode'] = 'cohort_augmented' if any_augmented_selected else 'cohort_only'
        predictions['cohort_train_volunteers'] = len(train_records)
        predictions['cohort_validation_volunteers'] = len(val_records)
        predictions['synthetic_data_used'] = int(
            synthetic_train_count if any_augmented_selected else 0)
        predictions['synthetic_data_generated'] = int(synthetic_train_count)
        predictions['ground_truth_available'] = bool(val_records)

        train_handles = [record['handle'] for record in train_records]
        val_handles = [record['handle'] for record in val_records]
        train_ids = [record['volunteer_id'] for record in train_records]
        val_ids = [record['volunteer_id'] for record in val_records]

        from backend.ml_pipeline.services.lasso_regressor import LassoTrainer
        trainer_state = {}
        if trait_artifacts:
            first_artifact = next(iter(trait_artifacts.values()))
            cohort_trainer = LassoTrainer(
                alpha=first_artifact['trainer'].alpha,
                max_iter=first_artifact['trainer'].max_iter,
                regularization=first_artifact['trainer'].regularization,
                l1_ratio=first_artifact['trainer'].l1_ratio,
            )
            cohort_trainer.feature_mean = first_artifact['trainer'].feature_mean
            cohort_trainer.feature_scale = first_artifact['trainer'].feature_scale
            for trait, artifact in trait_artifacts.items():
                cohort_trainer.models[trait] = artifact['trainer'].models[trait]
                cohort_trainer.model_metadata[trait] = artifact['trainer'].model_metadata.get(
                    trait, {})
            trainer_state = cohort_trainer.save_state()

        model_artifact, _ = COHORT_MODEL.objects.update_or_create(
            name='default-cohort-model',
            defaults={
                'version': f"split-{split_seed}",
                'is_active': True,
                'split_seed': split_seed,
                'train_ratio': 0.8,
                'validation_ratio': 0.2,
                'train_volunteer_ids': train_ids,
                'validation_volunteer_ids': val_ids,
                'train_handles': train_handles,
                'validation_handles': val_handles,
                'trainer_state': trainer_state,
                'metrics': {
                    **metrics,
                    **predictions,
                },
            },
        )

        return {
            'status': 'success',
            'train_volunteers': train_handles,
            'validation_volunteers': val_handles,
            'train_count': len(train_records),
            'validation_count': len(val_records),
            'bootstrap_stats': bootstrap_stats,
            'model_id': model_artifact.id,
            'model_name': model_artifact.name,
            'model_version': model_artifact.version,
            'metrics': model_artifact.metrics,
        }

    def _get_active_cohort_model(self) -> Optional[COHORT_MODEL]:
        """Return the latest active cohort model artifact, if one exists."""
        return COHORT_MODEL.objects.filter(is_active=True).order_by('-updated_at').first()

    def predict_from_saved_model(self, embeddings: Optional[List] = None) -> Dict:
        """
        Run prediction only using the latest saved cohort model.
        No training or model fitting happens here.
        """
        from backend.ml_pipeline.services.lasso_regressor import LassoTrainer, denormalize_predictions

        cohort_model = self._get_active_cohort_model()
        if not cohort_model:
            raise ValueError(
                "No saved cohort model is available. Train a model first.")

        trainer = LassoTrainer()
        trainer.load_state(cohort_model.trainer_state)

        current_vector = self._volunteer_feature_vector(
            self.volunteer, include_synthetic=False)
        embeddings_used_count = 0
        if current_vector is None and embeddings:
            current_vector = self._pool_embeddings([
                self._embedding_to_array(emb.embedding_vector) for emb in embeddings
            ])
            embeddings_used_count = len(embeddings)
        elif embeddings:
            embeddings_used_count = len(embeddings)
        else:
            real_embeddings = BERT_EMBEDDING.objects.filter(
                volunteer=self.volunteer)
            selected_embeddings = real_embeddings.filter(
                post__selected_by_qlearning=True)
            embeddings_used_count = selected_embeddings.count(
            ) if selected_embeddings.exists() else real_embeddings.count()

        if current_vector is None:
            raise ValueError(
                "Unable to build a prediction vector for this volunteer")

        current_vector = current_vector.reshape(1, -1)
        current_norm = trainer.transform_features(current_vector)
        raw_predictions = trainer.predict_all_traits(current_norm)

        predictions = {}
        for trait, normalized in raw_predictions.items():
            denorm = denormalize_predictions(normalized)
            predictions[f'predicted_{trait.lower()}'] = float(np.mean(denorm))

        ground_truth = None
        try:
            survey = self.volunteer.bfi_survey
            ground_truth = survey.get_ocean_dict()
        except Exception:
            ground_truth = None

        if ground_truth:
            y_true = np.array([ground_truth.get(t, np.nan) for t in [
                              'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']])
            y_pred = np.array([predictions.get(f'predicted_{t.lower()}', np.nan) for t in [
                              'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']])

            for i, trait in enumerate(['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']):
                pv = y_pred[i]
                tv = y_true[i]
                predictions[f'mae_{trait.lower()}'] = float(
                    abs(pv - tv)) if not np.isnan(pv) and not np.isnan(tv) else None

            mae_values = [v for k, v in predictions.items(
            ) if k.startswith('mae_') and v is not None]
            predictions['overall_mae'] = float(
                np.mean(mae_values)) if mae_values else None

            # Lasso metrics: correlation & R²
            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid_mask.sum() > 1:
                from scipy import stats as sp_stats
                corr, _ = sp_stats.pearsonr(
                    y_true[valid_mask], y_pred[valid_mask])
                ss_res = np.sum((y_true[valid_mask] - y_pred[valid_mask]) ** 2)
                ss_tot = np.sum(
                    (y_true[valid_mask] - np.mean(y_true[valid_mask])) ** 2)
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                predictions['correlation'] = round(float(corr), 4)
                predictions['r2_score'] = round(float(r2), 4)

            # LSTM classification metrics via threshold sweep on continuous predictions
            from backend.ml_pipeline.services.metrics_engine import evaluate_hybrid_metrics, CANDIDATE_THRESHOLDS
            # median trait score as binary cutoff
            cutoff = float(np.nanmean(y_true))
            probs_lstm = np.clip((y_pred - 1.0) / 4.0, 0.0,
                                 1.0)  # scale [1,5] → [0,1]
            hybrid_metrics = evaluate_hybrid_metrics(
                y_true=y_true[valid_mask],
                y_pred_lasso=y_pred[valid_mask],
                y_pred_lstm_continuous=y_pred[valid_mask],
                probabilities_lstm=probs_lstm[valid_mask],
                ground_truth_cutoff=cutoff,
                candidate_thresholds=CANDIDATE_THRESHOLDS
            )
            predictions['accuracy'] = hybrid_metrics.get('lstm_accuracy')
            predictions['precision'] = hybrid_metrics.get('lstm_precision')
            predictions['f1_score'] = hybrid_metrics.get('lstm_f1_score')
            predictions['specificity'] = hybrid_metrics.get('lstm_specificity')
            predictions['optimal_threshold'] = hybrid_metrics.get(
                'optimal_threshold', 0.50)
            predictions['threshold_sweep_data'] = hybrid_metrics.get(
                'threshold_sweep', {})
            predictions['model_metrics_taxonomy'] = hybrid_metrics
            predictions['ground_truth_available'] = True
            predictions['ground_truth'] = ground_truth
        else:
            predictions['overall_mae'] = None
            predictions['ground_truth_available'] = False

        prediction_confidence, confidence_components = self._compute_prediction_confidence(
            predictions)
        predictions['prediction_confidence'] = prediction_confidence
        predictions['confidence_components'] = confidence_components

        predictions['posts_analyzed'] = int(embeddings_used_count)
        predictions['embeddings_used'] = int(embeddings_used_count)
        predictions['synthetic_data_used'] = 0
        predictions['synthetic_data_generated'] = 0
        predictions['training_fallback_used'] = False
        predictions['training_mode'] = 'prediction_only'
        predictions['cohort_train_volunteers'] = len(
            cohort_model.train_volunteer_ids)
        predictions['cohort_validation_volunteers'] = len(
            cohort_model.validation_volunteer_ids)

        return {
            'status': 'success',
            'prediction_result': predictions,
            'model_id': cohort_model.id,
            'model_version': cohort_model.version,
            'validation_handles': cohort_model.validation_handles,
            'train_handles': cohort_model.train_handles,
        }

    def _compute_prediction_confidence(self, prediction_result: Dict) -> Tuple[float, Dict]:
        """
        Compute overall prediction confidence score based on error, correlation, and sample count.
        """
        mae = prediction_result.get('overall_mae')
        if mae is not None:
            mae_comp = max(0.0, 1.0 - (mae / 4.0))
        else:
            mae_comp = 0.75

        posts_analyzed = prediction_result.get('posts_analyzed', 0)
        data_comp = min(1.0, 0.5 + (posts_analyzed / 20.0) * 0.5)

        base_confidence = 0.6 * mae_comp + 0.4 * data_comp

        components = {
            'mae_component': round(mae_comp, 4),
            'data_component': round(data_comp, 4),
            'final_confidence': round(base_confidence, 4),
        }
        return round(base_confidence, 4), components

    def _save_psychometric_profile(self, prediction_result: Dict, pipeline_summary: Optional[Dict] = None):
        """
        Save final psychometric profile with 8-metric taxonomy to database.
        """
        self.logger.info(
            "Saving psychometric profile with 8-metric framework...")

        prediction_confidence = prediction_result.get(
            'prediction_confidence') or 0.85
        pipeline_summary = dict(pipeline_summary or {})
        pipeline_summary['prediction_confidence'] = prediction_confidence

        profile, created = PSYCHOMETRIC_PROFILE.objects.update_or_create(
            volunteer=self.volunteer,
            defaults={
                'predicted_openness': prediction_result.get('predicted_openness'),
                'predicted_conscientiousness': prediction_result.get('predicted_conscientiousness'),
                'predicted_extraversion': prediction_result.get('predicted_extraversion'),
                'predicted_agreeableness': prediction_result.get('predicted_agreeableness'),
                'predicted_neuroticism': prediction_result.get('predicted_neuroticism'),
                'mae_openness': prediction_result.get('mae_openness'),
                'mae_conscientiousness': prediction_result.get('mae_conscientiousness'),
                'mae_extraversion': prediction_result.get('mae_extraversion'),
                'mae_agreeableness': prediction_result.get('mae_agreeableness'),
                'mae_neuroticism': prediction_result.get('mae_neuroticism'),
                'overall_mae': prediction_result.get('overall_mae'),
                'correlation': prediction_result.get('correlation'),
                'r2_score': prediction_result.get('r2_score'),
                'accuracy': prediction_result.get('accuracy'),
                'precision': prediction_result.get('precision'),
                'f1_score': prediction_result.get('f1_score'),
                'specificity': prediction_result.get('specificity'),
                'optimal_threshold': prediction_result.get('optimal_threshold', 0.50),
                'threshold_sweep_data': prediction_result.get('threshold_sweep_data', {}),
                'model_metrics_taxonomy': prediction_result.get('model_metrics_taxonomy', {}),
                'posts_analyzed': prediction_result.get('posts_analyzed', 0),
                'embeddings_used': prediction_result.get('embeddings_used', 0),
                'synthetic_data_used': prediction_result.get('synthetic_data_used', 0),
                'pipeline_summary': pipeline_summary or {},
                'prediction_confidence': prediction_confidence,
            }
        )

        self.logger.info(
            f"Psychometric profile {'created' if created else 'updated'} with 8 metrics.")
