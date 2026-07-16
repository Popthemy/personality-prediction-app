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

logger = logging.getLogger('ml_pipeline')


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
            synthetic_data = self._step_4_gan_augmentation(selected_posts, embeddings)
            result['steps_completed'].append('gan_augmentation')
            result['metrics']['synthetic_samples'] = len(synthetic_data)
            
            # Step 5: Lasso training and prediction
            prediction_result = self._step_5_lasso_prediction(embeddings, synthetic_data)
            result['steps_completed'].append('lasso_prediction')
            result['metrics'].update(prediction_result)
            
            # Save psychometric profile
            self._save_psychometric_profile(prediction_result)
            result['steps_completed'].append('saved_profile')
            
            self.volunteer.pipeline_status = 'completed'
            self.volunteer.save()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            result['status'] = 'error'
            result['error'] = str(e)
            self.volunteer.pipeline_status = 'error'
            self.volunteer.save()
        
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

        preprocessor = TextPreprocessor()
        raw_posts = POST.objects.filter(volunteer=self.volunteer).order_by('-created_at_original')
        self.logger.info(f"Retrieved {raw_posts.count()} raw posts from database")

        posts = []
        for post in raw_posts:
            cleaned = preprocessor.clean(post.content)
            if preprocessor.is_valid(cleaned):
                post.cleaned_content = cleaned
                posts.append(post)
            else:
                self.logger.debug(f"Filtered out invalid/short post: {post.content[:50]!r}")

        self.logger.info(f"Filtered down to {len(posts)} valid posts after preprocessing")
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
        return embeddings
    
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
        import json
        
        # Get ground truth BFI scores
        bfi_survey = BFI_SURVEY.objects.filter(volunteer=self.volunteer).first()
        if not bfi_survey:
            raise ValueError("No BFI survey found for this volunteer")
        
        # Prepare training data
        X_train = []
        y_train_dict = {}
        
        # Real embeddings
        for emb in embeddings:
            if isinstance(emb.embedding_vector, str):
                embedding_list = json.loads(emb.embedding_vector)
            else:
                embedding_list = emb.embedding_vector
            X_train.append(embedding_list)
        
        # Synthetic embeddings
        for syn in synthetic_data:
            if isinstance(syn.synthetic_embedding, str):
                embedding_list = json.loads(syn.synthetic_embedding)
            else:
                embedding_list = syn.synthetic_embedding
            X_train.append(embedding_list)
        
        X_train = np.array(X_train)
        
        # Get OCEAN labels
        ground_truth = bfi_survey.get_ocean_dict()
        for trait, score in ground_truth.items():
            if score is not None:
                y_train_dict[trait] = np.full(len(X_train), score)
        
        # Train Lasso models (one per trait)
        trainer = LassoTrainer(alpha=0.001, max_iter=10000)
        predictions = {}
        metrics = {}
        
        for trait, y_values in y_train_dict.items():
            try:
                # Prepare data
                X_norm, y_norm = trainer.prepare_training_data(X_train, y_values)
                
                # Train
                trait_metrics = trainer.train_trait_model(X_norm, y_norm, trait)
                metrics[trait] = trait_metrics
                
                # Predict on full data
                y_pred_norm = trainer.predict_trait(trait, X_norm)
                y_pred = denormalize_predictions(y_pred_norm)
                
                # Use average prediction
                avg_pred = np.mean(y_pred)
                predictions[f'predicted_{trait.lower()}'] = float(avg_pred)
                
                # MAE against ground truth
                mae = np.abs(avg_pred - ground_truth[trait])
                predictions[f'mae_{trait.lower()}'] = float(mae)
                
                self.logger.info(f"{trait}: Pred={avg_pred:.2f}, Ground Truth={ground_truth[trait]:.2f}, MAE={mae:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {trait} model: {e}")
                predictions[f'predicted_{trait.lower()}'] = None
                predictions[f'mae_{trait.lower()}'] = None
        
        # Save Lasso models to DB
        for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
            if trait in trainer.models:
                coeff_json = trainer.get_all_coefficients(trait)
                
                LASSO_MODEL.objects.update_or_create(
                    volunteer=self.volunteer,
                    trait=trait.lower(),
                    defaults={
                        'alpha': 0.001,
                        'coefficients': coeff_json,
                        'intercept': float(trainer.models[trait].intercept_),
                        'training_samples_used': len(embeddings),
                        'synthetic_samples_used': len(synthetic_data),
                    }
                )
        
        # Calculate overall MAE
        mae_values = [v for k, v in predictions.items() if k.startswith('mae_') and v is not None]
        if mae_values:
            predictions['overall_mae'] = float(np.mean(mae_values))
        
        predictions['posts_analyzed'] = len(embeddings)
        predictions['embeddings_used'] = len(embeddings)
        predictions['synthetic_data_used'] = len(synthetic_data)
        
        self.logger.info(f"Lasso training complete. Overall MAE: {predictions.get('overall_mae', 'N/A')}")
        
        return predictions
    
    def _save_psychometric_profile(self, prediction_result: Dict):
        """
        Save final psychometric profile to database.
        
        Args:
            prediction_result: Dict from _step_5_lasso_prediction
        """
        self.logger.info("Saving psychometric profile...")
        
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
                'posts_analyzed': prediction_result.get('posts_analyzed', 0),
                'embeddings_used': prediction_result.get('embeddings_used', 0),
                'synthetic_data_used': prediction_result.get('synthetic_data_used', 0),
                'prediction_confidence': 0.85,  # Placeholder
            }
        )
        
        self.logger.info(f"Psychometric profile {'created' if created else 'updated'}")
