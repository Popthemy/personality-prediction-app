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
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from django.db import transaction
from backend.core.models import (
    VOLUNTEER, POST, BERT_EMBEDDING, Q_LEARNING_LOG,
    SYNTHETIC_DATA, LASSO_MODEL, PSYCHOMETRIC_PROFILE, BFI_SURVEY
)
from backend.ml_pipeline.services.timeline_exporter import export_cleaned_posts_to_txt

logger = logging.getLogger('ml_pipeline')

SYNTHETIC_SAMPLE_WEIGHT = 0.35


class PipelineOrchestrator:
    """Main pipeline orchestrator."""
    
    def __init__(self, volunteer_id: int):
        """
        Initialize orchestrator for a volunteer.
        
        Args:
            volunteer_id: Volunteer database ID
        """
        self.volunteer = VOLUNTEER.objects.get(id=volunteer_id)
        self.logger = logging.getLogger(f'ml_pipeline.{self.volunteer.x_handle}')

    def _set_pipeline_status(self, status: str):
        """Persist the volunteer pipeline status."""
        self.volunteer.pipeline_status = status
        self.volunteer.save(update_fields=['pipeline_status'])

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return float(max(minimum, min(maximum, value)))

    def _compute_prediction_confidence(self, prediction_result: Dict) -> tuple[float, Dict[str, float]]:
        """
        Derive a reproducible confidence score from the actual run metrics.

        The score blends:
        - prediction error (lower MAE increases confidence)
        - agreement with ground truth (correlation / R²)
        - data coverage (more analyzed posts and synthetic samples)
        - a small penalty when we had to fall back to a tiny labeled cohort
        """
        overall_mae = prediction_result.get('overall_mae')
        correlation = prediction_result.get('correlation')
        r2_score_value = prediction_result.get('r2_score')
        posts_analyzed = int(prediction_result.get('posts_analyzed', 0) or 0)
        synthetic_used = int(prediction_result.get('synthetic_data_used', 0) or 0)
        fallback_used = bool(prediction_result.get('training_fallback_used', False))

        mae_component = self._clamp(1.0 - (float(overall_mae) / 4.0)) if overall_mae is not None else 0.5
        corr_component = self._clamp(((float(correlation) + 1.0) / 2.0)) if correlation is not None else 0.5
        r2_component = self._clamp(((float(r2_score_value) + 1.0) / 2.0)) if r2_score_value is not None else 0.5
        data_component = self._clamp((posts_analyzed / 10.0) * 0.7 + (synthetic_used / 10.0) * 0.3)
        fallback_multiplier = 0.85 if fallback_used else 1.0

        confidence = (
            0.35 * mae_component
            + 0.25 * corr_component
            + 0.20 * r2_component
            + 0.20 * data_component
        ) * fallback_multiplier

        confidence = self._clamp(confidence)
        components = {
            'mae_component': round(mae_component, 4),
            'correlation_component': round(corr_component, 4),
            'r2_component': round(r2_component, 4),
            'data_component': round(data_component, 4),
            'fallback_multiplier': round(fallback_multiplier, 4),
        }
        return confidence, components

    def run_qlearning_phase(self) -> Dict:
        """Run the Q-Learning selection phase only."""
        try:
            self._set_pipeline_status('processing')
            posts = self._step_1_input_data()
            selected_posts = self._step_2_qlearning_selection(posts)
            return {
                'status': 'success',
                'phase': 'qlearning',
                'input_posts': len(posts),
                'selected_posts': len(selected_posts),
            }
        except Exception as e:
            self._set_pipeline_status('error')
            return {
                'status': 'error',
                'phase': 'qlearning',
                'error': str(e),
            }

    def run_bert_phase(self) -> Dict:
        """Run the BERT embedding phase only."""
        try:
            self._set_pipeline_status('processing')
            selected_posts = list(
                POST.objects.filter(volunteer=self.volunteer, selected_by_qlearning=True)
                .order_by('-created_at_original')
            )
            if not selected_posts:
                raise ValueError("Run Q-Learning first to select posts for embedding")

            embeddings = self._step_3_bert_embedding(selected_posts)
            return {
                'status': 'success',
                'phase': 'bert',
                'selected_posts': len(selected_posts),
                'embeddings_created': len(embeddings),
            }
        except Exception as e:
            self._set_pipeline_status('error')
            return {
                'status': 'error',
                'phase': 'bert',
                'error': str(e),
            }

    def run_gan_phase(self) -> Dict:
        """Run the GAN augmentation phase only."""
        try:
            self._set_pipeline_status('processing')
            selected_posts = list(
                POST.objects.filter(volunteer=self.volunteer, selected_by_qlearning=True)
                .order_by('-created_at_original')
            )
            embeddings = list(
                BERT_EMBEDDING.objects.filter(post__in=selected_posts).order_by('created_at')
            )
            if not embeddings:
                raise ValueError("Run BERT embedding first to create embeddings for augmentation")

            synthetic_data = self._step_4_gan_augmentation(selected_posts, embeddings)
            return {
                'status': 'success',
                'phase': 'gan',
                'synthetic_samples': len(synthetic_data),
            }
        except Exception as e:
            self._set_pipeline_status('error')
            return {
                'status': 'error',
                'phase': 'gan',
                'error': str(e),
            }

    def run_lasso_phase(self) -> Dict:
        """Run the Lasso training/prediction phase only."""
        try:
            self._set_pipeline_status('processing')
            selected_posts = list(
                POST.objects.filter(volunteer=self.volunteer, selected_by_qlearning=True)
                .order_by('-created_at_original')
            )
            embeddings = list(
                BERT_EMBEDDING.objects.filter(post__in=selected_posts).order_by('created_at')
            )
            synthetic_data = list(
                SYNTHETIC_DATA.objects.filter(volunteer=self.volunteer, used_in_training=True).order_by('created_at')
            )

            if not embeddings:
                raise ValueError("Run BERT embedding first before Lasso training")

            prediction_result = self._step_5_lasso_prediction(embeddings, synthetic_data)
            self._save_psychometric_profile(prediction_result)
            self._set_pipeline_status('completed')

            return {
                'status': 'success',
                'phase': 'lasso',
                **prediction_result,
            }
        except Exception as e:
            self._set_pipeline_status('error')
            return {
                'status': 'error',
                'phase': 'lasso',
                'error': str(e),
            }
    
    def run_full_pipeline(self) -> Dict:
        """
        Execute complete pipeline: Input → Q-Learn → BERT → GAN → Lasso.
        
        Returns:
            Result dict with status and metrics
        """
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING FULL PIPELINE for {self.volunteer.x_handle}")
        self.logger.info("=" * 80)
        
        result = {
            'volunteer_id': self.volunteer.id,
            'volunteer_handle': self.volunteer.x_handle,
            'status': 'success',
            'steps_completed': [],
            'metrics': {},
            'error': None,
        }
        
        try:
            # Step 1: Input data (posts already fetched, stored in DB)
            posts = self._step_1_input_data()
            result['steps_completed'].append('input_data')
            result['metrics']['input_posts'] = len(posts)
            
            if len(posts) == 0:
                raise ValueError("No posts available for processing")
            
            # Step 2: Q-Learning selection
            selected_posts = self._step_2_qlearning_selection(posts)
            result['steps_completed'].append('qlearning_selection')
            result['metrics']['selected_posts'] = len(selected_posts)
            
            # Step 3: BERT embeddings
            embeddings = self._step_3_bert_embedding(selected_posts)
            result['steps_completed'].append('bert_embedding')
            result['metrics']['embeddings_created'] = len(embeddings)
            
            # Step 4: GAN augmentation (if training)
            prior_synthetic_count = SYNTHETIC_DATA.objects.filter(
                volunteer=self.volunteer,
                used_in_training=True,
            ).count()
            synthetic_data = self._step_4_gan_augmentation(selected_posts, embeddings)
            result['steps_completed'].append('gan_augmentation')
            result['metrics']['synthetic_samples'] = len(synthetic_data)
            result['metrics']['synthetic_samples_saved'] = len(synthetic_data)
            result['metrics']['synthetic_samples_reused'] = prior_synthetic_count
            result['metrics']['training_mode'] = (
                'retrain' if prior_synthetic_count > 0 else 'fresh_train'
            )
            
            # Step 5: Lasso training and prediction
            prediction_result = self._step_5_lasso_prediction(embeddings, synthetic_data)
            result['steps_completed'].append('lasso_prediction')
            result['metrics'].update(prediction_result)
            
            # Save psychometric profile
            self._save_psychometric_profile(prediction_result, result['metrics'])
            result['steps_completed'].append('saved_profile')
            
            self._set_pipeline_status('completed')
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            result['status'] = 'error'
            result['error'] = str(e)
            self._set_pipeline_status('error')
        
        self.logger.info(f"Pipeline result: {result['status']}")
        return result
    
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
        raw_posts = POST.objects.filter(volunteer=self.volunteer).order_by('-created_at_original')
        self.logger.info(f"Retrieved {raw_posts.count()} raw posts from database")

        if raw_posts.count() == 0:
            self.logger.info("No posts in database. Attempting live X fetch before aborting.")
            fetcher = TwitterFetcher()
            saved, skipped = fetcher.fetch_and_save(self.volunteer)
            self.logger.info(
                f"Live fetch attempt for @{self.volunteer.x_handle} returned saved={saved}, skipped={skipped}"
            )
            raw_posts = POST.objects.filter(volunteer=self.volunteer).order_by('-created_at_original')
            self.logger.info(f"Database now has {raw_posts.count()} posts after fetch attempt")

        posts = []
        for post in raw_posts:
            cleaned = preprocessor.clean(post.content)
            if preprocessor.is_valid(cleaned):
                post.cleaned_content = cleaned
                posts.append(post)
            else:
                self.logger.debug(f"Filtered out invalid/short post: {post.content[:50]!r}")

        self.logger.info(f"Filtered down to {len(posts)} valid posts after preprocessing")
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
        Step 2: Q-Learning active signal selection.
        
        Args:
            posts: List of POST objects
        
        Returns:
            List of selected POST objects
        """
        self.logger.info("STEP 2: Q-Learning Active Signal Selection")
        
        from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, create_post_features
        
        agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.05)
        
        # Create feature dicts
        post_features = []
        for post in posts:
            features = create_post_features(post)
            post_features.append({
                'id': post.id,
                'content': getattr(post, 'cleaned_content', post.content),
                **features,
            })
        
        # Select top posts
        top_k = min(10, len(posts))  # Default: select top 10 posts
        selected = agent.select_posts(post_features, top_k=top_k, training=False)
        
        # Log decisions and mark posts
        selected_post_ids = {s['id'] for s in selected}
        selected_posts = []
        
        for post in posts:
            if post.id in selected_post_ids:
                post.selected_by_qlearning = True
                
                # Find Q-value
                for s in selected:
                    if s['id'] == post.id:
                        post.q_value = s.get('q_value', 0)
                        break
                selected_posts.append(post)
            post.save()
        
        self.logger.info(f"Q-Learning selected {len(selected_posts)} posts")
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
        verification = self._verify_bert_embedding_persistence(posts, embeddings)
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
        
        self.logger.info(f"Generated {len(synthetic_list)} synthetic training samples")
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

        vectors = [self._embedding_to_array(value) for value in embedding_values]
        return np.mean(np.vstack(vectors), axis=0)

    def _volunteer_feature_vector(self, volunteer: VOLUNTEER, include_synthetic: bool = True) -> Optional[np.ndarray]:
        """
        Build a volunteer-level feature vector using pooled embeddings.

        Real post embeddings are pooled first. Synthetic embeddings are appended
        as additional samples when requested to stabilize the representation.
        """
        real_embeddings = BERT_EMBEDDING.objects.filter(volunteer=volunteer)
        selected_embeddings = real_embeddings.filter(post__selected_by_qlearning=True)
        if selected_embeddings.exists():
            real_embeddings = selected_embeddings

        vectors = [self._embedding_to_array(emb.embedding_vector) for emb in real_embeddings]

        if include_synthetic:
            synthetic_rows = SYNTHETIC_DATA.objects.filter(volunteer=volunteer, used_in_training=True)
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
            base_vector = self._volunteer_feature_vector(volunteer, include_synthetic=False)
            if base_vector is None:
                continue

            survey = volunteer.bfi_survey
            ocean = survey.get_ocean_dict()

            samples.append(base_vector)
            for trait, value in ocean.items():
                trait_labels[trait].append(value)

            synthetic_rows = SYNTHETIC_DATA.objects.filter(volunteer=volunteer, used_in_training=True)
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
            base_vector = self._volunteer_feature_vector(volunteer, include_synthetic=False)
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
                    samples.append(self._embedding_to_array(row.synthetic_embedding))
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

    def _step_5_lasso_prediction(self, embeddings: List, synthetic_data: List) -> Dict:
        """
        Step 5: Lasso regression training and personality prediction.
        
        Args:
            embeddings: List of BERT_EMBEDDING objects
            synthetic_data: List of SYNTHETIC_DATA objects
        
        Returns:
            Dict with predictions and metrics
        """
        self.logger.info("STEP 5: Lasso Regression Training & Personality Prediction")
        
        from backend.ml_pipeline.services.lasso_regressor import LassoTrainer, denormalize_predictions
        from sklearn.model_selection import train_test_split

        # Get ground truth BFI scores
        bfi_survey = BFI_SURVEY.objects.filter(volunteer=self.volunteer).first()
        if not bfi_survey:
            raise ValueError("No BFI survey found for this volunteer")

        ground_truth = bfi_survey.get_ocean_dict()
        self.logger.info(
            "BFI ground truth found for @%s: O=%.2f C=%.2f E=%.2f A=%.2f N=%.2f",
            self.volunteer.x_handle,
            ground_truth.get('Openness') or 0.0,
            ground_truth.get('Conscientiousness') or 0.0,
            ground_truth.get('Extraversion') or 0.0,
            ground_truth.get('Agreeableness') or 0.0,
            ground_truth.get('Neuroticism') or 0.0,
        )

        # Build a volunteer-level cohort from other labeled volunteers only.
        labeled_records = self._build_labeled_volunteer_records(
            exclude_volunteer_ids=[self.volunteer.id]
        )
        if not labeled_records:
            raise ValueError("No cohort volunteers with BFI survey found for training")

        if len(labeled_records) >= 3:
            val_size = max(1, int(round(len(labeled_records) * 0.2)))
            if val_size >= len(labeled_records):
                val_size = 1
            train_records, val_records = train_test_split(
                labeled_records,
                test_size=val_size,
                random_state=42,
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
            raise ValueError("No training samples available after cohort assembly")

        X_val_raw = None
        y_val_dict = {}
        if val_records:
            X_val_raw = np.vstack([record['vector'] for record in val_records])
            y_val_dict = {
                trait: np.array([record['labels'][trait] for record in val_records], dtype=float)
                for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            }

        fallback_used = False
        self.logger.info(
            "Cohort training assembled with %s train volunteers and %s validation volunteers",
            len(train_records),
            len(val_records),
        )

        predictions = {}
        metrics = {}

        current_vector = self._volunteer_feature_vector(self.volunteer, include_synthetic=True)
        if current_vector is None and embeddings:
            current_vector = self._pool_embeddings(
                [self._embedding_to_array(emb.embedding_vector) for emb in embeddings]
            )

        if current_vector is None:
            raise ValueError("Unable to build a feature vector for the current volunteer")

        current_vector = current_vector.reshape(1, -1)

        trait_artifacts = {}
        selected_synthetic_count = 0
        chosen_variants = {}
        can_validate = bool(val_records and X_val_raw is not None and len(X_val_raw) > 0)

        def _fit_variant(trait_name: str, X_train: np.ndarray, y_train: np.ndarray,
                         sample_weight: Optional[np.ndarray], variant_name: str):
            variant_trainer = LassoTrainer(
                alpha=0.001,
                max_iter=10000,
                regularization='elasticnet',
                l1_ratio=0.5,
            )
            X_fit, y_fit_norm = variant_trainer.prepare_training_data(X_train, y_train)
            X_val_norm, y_val_norm = None, None
            if can_validate and trait_name in y_val_dict and len(y_val_dict[trait_name]) > 0:
                X_val_norm = variant_trainer.transform_features(X_val_raw)
                y_val_norm = (y_val_dict[trait_name] - 1.0) / 4.0

            trait_metrics = variant_trainer.train_trait_model(
                X_fit,
                y_fit_norm,
                trait_name,
                validate_X=X_val_norm,
                validate_y=y_val_norm,
                sample_weight=sample_weight,
            )

            current_norm = variant_trainer.transform_features(current_vector)
            y_pred_norm = variant_trainer.predict_trait(trait_name, current_norm)
            y_pred = denormalize_predictions(y_pred_norm)
            avg_pred = float(np.mean(y_pred))
            val_mae = trait_metrics.get('validation_mae')
            if val_mae is None:
                val_mae = trait_metrics.get('train_mae', float('inf'))

            return {
                'trainer': variant_trainer,
                'metrics': trait_metrics,
                'prediction': avg_pred,
                'validation_mae': float(val_mae),
                'variant': variant_name,
                'training_samples_used': int(X_train.shape[0]),
            }

        any_augmented_selected = False
        for trait, y_values in real_y_train_dict.items():
            if len(real_X_train) == 0 or len(y_values) == 0:
                predictions[f'predicted_{trait.lower()}'] = None
                predictions[f'mae_{trait.lower()}'] = None
                continue

            try:
                baseline_result = _fit_variant(
                    trait,
                    real_X_train,
                    y_values,
                    np.ones(len(real_X_train), dtype=float),
                    'real_only',
                )

                chosen_result = baseline_result
                if can_validate:
                    augmented_result = _fit_variant(
                        trait,
                        augmented_X_train,
                        augmented_y_train_dict[trait],
                        augmented_weights,
                        'augmented',
                    )
                    if augmented_result['validation_mae'] <= baseline_result['validation_mae']:
                        chosen_result = augmented_result

                trait_artifacts[trait] = chosen_result
                metrics[trait] = chosen_result['metrics']
                if chosen_result['variant'] == 'augmented':
                    any_augmented_selected = True
                chosen_variants[trait] = chosen_result['variant']

                avg_pred = chosen_result['prediction']
                predictions[f'predicted_{trait.lower()}'] = avg_pred

                if ground_truth.get(trait) is not None:
                    mae = abs(avg_pred - ground_truth[trait])
                    predictions[f'mae_{trait.lower()}'] = float(mae)
                    self.logger.info(
                        f"{trait}: Pred={avg_pred:.2f}, Ground Truth={ground_truth[trait]:.2f}, MAE={mae:.4f}"
                    )
                else:
                    predictions[f'mae_{trait.lower()}'] = None

            except Exception as e:
                self.logger.error(f"Error training {trait} model: {e}")
                predictions[f'predicted_{trait.lower()}'] = None
                predictions[f'mae_{trait.lower()}'] = None

        for trait, artifact in trait_artifacts.items():
            trainer = artifact['trainer']
            coeff_json = trainer.get_all_coefficients(trait)
            model_obj, _ = LASSO_MODEL.objects.update_or_create(
                volunteer=self.volunteer,
                trait=trait.lower(),
                defaults={
                    'alpha': trainer.alpha,
                    'coefficients': coeff_json,
                    'intercept': float(trainer.models[trait].intercept_),
                    'training_samples_used': int(artifact['training_samples_used']),
                    'synthetic_samples_used': int(synthetic_train_count if artifact['variant'] == 'augmented' else 0),
                }
            )

            # Persist validation metrics when available.
            trait_metrics = artifact['metrics']
            if trait_metrics:
                for key in ['train_mae', 'train_rmse', 'train_r2', 'validation_mae', 'validation_rmse', 'validation_r2']:
                    if key in trait_metrics:
                        setattr(model_obj, key, trait_metrics[key])
                model_obj.save()

        selected_synthetic_count = synthetic_train_count if any_augmented_selected else 0

        mae_values = [v for k, v in predictions.items() if k.startswith('mae_') and v is not None]
        if mae_values:
            predictions['overall_mae'] = float(np.mean(mae_values))
        else:
            predictions['overall_mae'] = None

        predictions['posts_analyzed'] = len(embeddings)
        predictions['embeddings_used'] = len(embeddings)
        predictions['synthetic_data_used'] = selected_synthetic_count
        predictions['synthetic_data_generated'] = len(synthetic_data)
        predictions['training_fallback_used'] = fallback_used
        predictions['training_mode'] = 'cohort_augmented' if any_augmented_selected else 'cohort_only'
        predictions['cohort_train_volunteers'] = len(train_records)
        predictions['cohort_validation_volunteers'] = len(val_records)
        predictions['ground_truth_available'] = True
        predictions['ground_truth'] = ground_truth

        paired_ground_truth = []
        paired_predictions = []
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            predicted_value = predictions.get(f'predicted_{trait.lower()}')
            ground_truth_value = ground_truth.get(trait)
            if predicted_value is None or ground_truth_value is None:
                continue
            paired_predictions.append(float(predicted_value))
            paired_ground_truth.append(float(ground_truth_value))

        if len(paired_ground_truth) >= 2:
            try:
                from scipy.stats import pearsonr
                from sklearn.metrics import r2_score

                predictions['correlation'] = float(pearsonr(paired_ground_truth, paired_predictions)[0])
                predictions['r2_score'] = float(r2_score(paired_ground_truth, paired_predictions))
            except Exception as metric_error:
                self.logger.warning(f"Unable to compute correlation/R2: {metric_error}")
                predictions['correlation'] = None
                predictions['r2_score'] = None
        else:
            predictions['correlation'] = None
            predictions['r2_score'] = None

        self.logger.info(
            f"Lasso training complete. Overall MAE: {predictions.get('overall_mae', 'N/A')}"
        )

        return predictions
    
    def _save_psychometric_profile(self, prediction_result: Dict, pipeline_summary: Optional[Dict] = None):
        """
        Save final psychometric profile to database.
        
        Args:
            prediction_result: Dict from _step_5_lasso_prediction
            pipeline_summary: Dict of pipeline run metadata
        """
        self.logger.info("Saving psychometric profile...")

        prediction_confidence, confidence_components = self._compute_prediction_confidence(prediction_result)
        pipeline_summary = dict(pipeline_summary or {})
        pipeline_summary['confidence_components'] = confidence_components
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
                'posts_analyzed': prediction_result.get('posts_analyzed', 0),
                'embeddings_used': prediction_result.get('embeddings_used', 0),
                'synthetic_data_used': prediction_result.get('synthetic_data_used', 0),
                'pipeline_summary': pipeline_summary or {},
                'prediction_confidence': prediction_confidence,
            }
        )
        
        self.logger.info(f"Psychometric profile {'created' if created else 'updated'}")
