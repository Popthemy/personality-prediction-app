"""Tests for the ML pipeline helpers and trainer behavior."""

import numpy as np
from pathlib import Path
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from backend.core.models import VOLUNTEER, POST, BERT_EMBEDDING
from backend.ml_pipeline.services.insight_engine import build_domain_insights
from backend.ml_pipeline.services.lasso_regressor import LassoTrainer
from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
from backend.ml_pipeline.services.timeline_exporter import export_cleaned_posts_to_txt, sanitize_handle


class PipelineConfidenceTests(SimpleTestCase):
    def test_prediction_confidence_reflects_metrics_and_bounds(self):
        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        prediction_result = {
            'overall_mae': 0.5,
            'correlation': 0.8,
            'r2_score': 0.6,
            'posts_analyzed': 12,
            'synthetic_data_used': 8,
            'training_fallback_used': False,
        }

        confidence, components = orchestrator._compute_prediction_confidence(prediction_result)

        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIn('mae_component', components)
        self.assertIn('data_component', components)


class LassoTrainerTests(SimpleTestCase):
    def test_weighted_elasticnet_training_runs(self):
        rng = np.random.RandomState(7)
        X = rng.normal(size=(8, 6))
        y = np.array([1.2, 1.5, 2.0, 2.6, 3.1, 3.6, 4.1, 4.5], dtype=float)
        weights = np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.4], dtype=float)

        trainer = LassoTrainer(regularization='elasticnet', l1_ratio=0.5)
        X_fit, y_fit = trainer.prepare_training_data(X, y)
        metrics = trainer.train_trait_model(
            X_fit,
            y_fit,
            'Openness',
            sample_weight=weights,
        )

        self.assertTrue(metrics['sample_weight_used'])
        self.assertEqual(metrics['trait'], 'Openness')
        self.assertGreaterEqual(metrics['training_samples'], 8)
        prediction = trainer.predict_trait('Openness', trainer.transform_features(X[:1]))
        self.assertEqual(prediction.shape[0], 1)


class InsightEngineTests(SimpleTestCase):
    def test_domain_insights_are_dynamic(self):
        prediction = {
            'predicted_openness': 4.2,
            'predicted_conscientiousness': 3.9,
            'predicted_extraversion': 2.7,
            'predicted_agreeableness': 3.1,
            'predicted_neuroticism': 2.2,
            'overall_mae': 0.41,
            'correlation': 0.78,
            'r2_score': 0.52,
            'posts_analyzed': 18,
            'embeddings_used': 18,
            'synthetic_data_used': 10,
            'prediction_confidence': 0.84,
            'training_mode': 'cohort_augmented',
            'cohort_validation_volunteers': 2,
        }

        insights = build_domain_insights(prediction_result=prediction)

        self.assertIn('education', insights)
        self.assertIn('responsible_ai', insights)
        self.assertIn('summary', insights)
        self.assertIn('high', insights['education']['summary'].lower())
        self.assertGreater(len(insights['responsible_ai']['signals']), 0)


class BertPersistenceTests(TestCase):
    def test_bert_persistence_verification_matches_database_rows(self):
        user = get_user_model().objects.create_user(
            username='researcher-bert-test',
            email='bert-test@example.com',
            password='testpass123',
        )
        volunteer = VOLUNTEER.objects.create(
            x_handle='bert_check_user',
            researcher=user,
            consent_given=True,
            pipeline_status='processing',
        )

        post1 = POST.objects.create(
            volunteer=volunteer,
            x_post_id='post-1',
            content='First test post',
            created_at_original=timezone.now(),
            selected_by_qlearning=True,
        )
        post2 = POST.objects.create(
            volunteer=volunteer,
            x_post_id='post-2',
            content='Second test post',
            created_at_original=timezone.now(),
            selected_by_qlearning=True,
        )

        embeddings = [
            BERT_EMBEDDING.objects.create(
                post=post1,
                volunteer=volunteer,
                embedding_vector={'0': 0.1, '1': 0.2},
                model_name='bert-base-uncased',
            ),
            BERT_EMBEDDING.objects.create(
                post=post2,
                volunteer=volunteer,
                embedding_vector={'0': 0.3, '1': 0.4},
                model_name='bert-base-uncased',
            ),
        ]

        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator.volunteer = volunteer

        verification = orchestrator._verify_bert_embedding_persistence([post1, post2], embeddings)

        self.assertEqual(verification['expected_count'], 2)
        self.assertEqual(verification['created_count'], 2)
        self.assertEqual(verification['persisted_count'], 2)


class TimelineExportTests(SimpleTestCase):
    def test_handle_sanitization_and_export(self):
        self.assertEqual(sanitize_handle('@sonofgod0199'), 'sonofgod0199')
        self.assertEqual(sanitize_handle('  @bad handle!  '), 'bad_handle_')

        export_path = export_cleaned_posts_to_txt(
            '@sonofgod0199',
            ['First cleaned post', 'Second cleaned post'],
        )

        self.assertTrue(Path(export_path).exists())
        self.assertTrue(str(export_path).endswith('sonofgod0199.txt'))
        content = Path(export_path).read_text(encoding='utf-8')
        self.assertIn('1. First cleaned post', content)
        self.assertIn('2. Second cleaned post', content)
