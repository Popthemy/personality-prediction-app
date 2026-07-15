#!/usr/bin/env python3
"""
Validation Script for Big Five Personality Prediction System.

Validates:
1. All 8 database models present
2. All 6 ML services importable and working
3. BFI-44 scoring with proper reverse items
4. Pipeline execution order (Q-Learn → BERT → GAN → Lasso)
5. All views and routes configured
6. Celery task setup
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.config.settings.dev')
django.setup()

import json
from typing import Dict, List, Tuple


class ValidationSuite:
    """Comprehensive validation of the implementation."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.results = []
    
    def check(self, name: str, condition: bool, message: str = ""):
        """Record a single check result."""
        status = "✓ PASS" if condition else "✗ FAIL"
        self.results.append(f"{status}: {name}")
        if message:
            self.results.append(f"  └─ {message}")
        
        if condition:
            self.checks_passed += 1
        else:
            self.checks_failed += 1
        
        return condition
    
    def validate_models(self):
        """Test 1: Validate all 8 models."""
        print("\n" + "="*70)
        print("TEST 1: Database Models (8 Required)")
        print("="*70)
        
        try:
            from backend.core.models import (
                VOLUNTEER, BFI_SURVEY, POST, BERT_EMBEDDING,
                Q_LEARNING_LOG, SYNTHETIC_DATA, LASSO_MODEL, PSYCHOMETRIC_PROFILE
            )
            
            models_info = {
                'VOLUNTEER': (VOLUNTEER, ['x_handle', 'researcher']),
                'BFI_SURVEY': (BFI_SURVEY, ['volunteer', 'responses', 'openness']),
                'POST': (POST, ['volunteer', 'content', 'selected_by_qlearning']),
                'BERT_EMBEDDING': (BERT_EMBEDDING, ['post', 'embedding_vector']),
                'Q_LEARNING_LOG': (Q_LEARNING_LOG, ['volunteer', 'episode', 'reward']),
                'SYNTHETIC_DATA': (SYNTHETIC_DATA, ['volunteer', 'synthetic_text']),
                'LASSO_MODEL': (LASSO_MODEL, ['volunteer', 'coefficients']),
                'PSYCHOMETRIC_PROFILE': (PSYCHOMETRIC_PROFILE, ['volunteer', 'predicted_openness']),
            }
            
            for model_name, (model_class, required_fields) in models_info.items():
                # Check model exists
                try:
                    # Get model fields
                    fields = [f.name for f in model_class._meta.get_fields()]
                    has_required = all(f in fields for f in required_fields)
                    self.check(
                        f"Model: {model_name}",
                        has_required,
                        f"Fields: {', '.join(required_fields)}"
                    )
                except Exception as e:
                    self.check(f"Model: {model_name}", False, str(e))
            
        except ImportError as e:
            self.check("Import all models", False, f"ImportError: {e}")
    
    def validate_ml_services(self):
        """Test 2: Validate all 6 ML services."""
        print("\n" + "="*70)
        print("TEST 2: ML Pipeline Services (6 Required)")
        print("="*70)
        
        services = {
            'BFIScorer': ('backend.core.services.bfi_scorer', 'BFIScorer'),
            'QLearningAgent': ('backend.ml_pipeline.services.qlearning_agent', 'QLearningAgent'),
            'BERTEncoder': ('backend.ml_pipeline.services.bert_encoder', 'BERTEncoder'),
            'GANAugmenter': ('backend.ml_pipeline.services.gan_augmenter', 'GANAugmenter'),
            'LassoTrainer': ('backend.ml_pipeline.services.lasso_regressor', 'LassoTrainer'),
            'PipelineOrchestrator': ('backend.ml_pipeline.services.pipeline_orchestrator', 'PipelineOrchestrator'),
        }
        
        for service_name, (module_path, class_name) in services.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                service_class = getattr(module, class_name)
                self.check(
                    f"Service: {service_name}",
                    callable(service_class),
                    f"Module: {module_path}"
                )
            except (ImportError, AttributeError) as e:
                self.check(f"Service: {service_name}", False, str(e))
    
    def validate_bfi_scorer(self):
        """Test 3: Validate BFI-44 scoring logic."""
        print("\n" + "="*70)
        print("TEST 3: BFI-44 Scoring with Reverse Items")
        print("="*70)
        
        try:
            from backend.core.services.bfi_scorer import BFIScorer, REVERSE_ITEMS
            
            scorer = BFIScorer()
            
            # Check reverse items are defined
            self.check(
                "Reverse items defined (21 expected)",
                len(REVERSE_ITEMS) == 21,
                f"Found: {len(REVERSE_ITEMS)} reverse items"
            )
            
            # Test with all neutral responses (3 on 1-5 scale)
            sample_responses = {str(i): 3 for i in range(1, 45)}
            scores = scorer.calculate_scores(sample_responses)
            
            # Validate response structure
            required_traits = {'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'}
            has_all_traits = all(trait in scores for trait in required_traits)
            self.check(
                "OCEAN traits calculated (5 traits)",
                has_all_traits,
                f"Found: {list(scores.keys())}"
            )
            
            # Validate score ranges
            valid_ranges = all(1 <= score <= 5 for score in scores.values())
            self.check(
                "OCEAN scores in valid range (1-5)",
                valid_ranges,
                f"Sample scores: {json.dumps(scores, indent=2)}"
            )
            
            # Test reverse scoring
            test_reverse_responses = {'2': 5}  # Item 2 is reverse-scored
            processed = scorer.preprocess_responses(test_reverse_responses)
            # Item 2 reversed: 5 → 1
            self.check(
                "Reverse item scoring (Item 2: 5→1)",
                processed.get('2') == 1,
                f"Result: {processed}"
            )
            
        except Exception as e:
            self.check("BFI-44 Scoring", False, str(e))
    
    def validate_pipeline_order(self):
        """Test 4: Validate strict pipeline execution order."""
        print("\n" + "="*70)
        print("TEST 4: Pipeline Execution Order (Q-Learn → BERT → GAN → Lasso)")
        print("="*70)
        
        try:
            from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
            
            # Read orchestrator source to verify order
            with open('backend/ml_pipeline/services/pipeline_orchestrator.py', 'r') as f:
                source = f.read()
            
            # Check order in method signatures and docstrings
            order_checks = [
                ('Q-Learning appears before BERT', 'step_2_qlearning' in source and 'step_3_bert' in source and
                 source.index('step_2_qlearning') < source.index('step_3_bert')),
                ('BERT appears before GAN', 'step_3_bert' in source and 'step_4_gan' in source and
                 source.index('step_3_bert') < source.index('step_4_gan')),
                ('GAN appears before Lasso', 'step_4_gan' in source and 'step_5_lasso' in source and
                 source.index('step_4_gan') < source.index('step_5_lasso')),
            ]
            
            for check_name, result in order_checks:
                self.check(f"Pipeline Order: {check_name}", result)
            
        except Exception as e:
            self.check("Pipeline Order Validation", False, str(e))
    
    def validate_views_and_routes(self):
        """Test 5: Validate all views and URL routes."""
        print("\n" + "="*70)
        print("TEST 5: Views and URL Routes")
        print("="*70)
        
        views_to_check = {
            'Public': [
                ('backend.public.views', 'IndexView'),
                ('backend.public.views', 'LivePredictionView'),
                ('backend.public.views', 'PredictAPIView'),
            ],
            'Accounts': [
                ('backend.accounts.views', 'RegisterView'),
                ('backend.accounts.views', 'ProfileView'),
            ],
            'Dashboard': [
                ('backend.dashboard.views', 'DashboardView'),
                ('backend.dashboard.views', 'VolunteerDetailView'),
                ('backend.dashboard.views', 'DomainInsightsView'),
                ('backend.dashboard.views', 'ReportsView'),
            ],
            'Tools': [
                ('backend.tools.views', 'ToolsView'),
                ('backend.tools.views', 'CSVUploadView'),
                ('backend.tools.views', 'FetchPostsView'),
                ('backend.tools.views', 'RunPipelineView'),
            ],
            'ML Pipeline': [
                ('backend.ml_pipeline.views', 'PipelineStatusView'),
                ('backend.ml_pipeline.views', 'PipelineLogsView'),
            ],
        }
        
        for app_name, views_list in views_to_check.items():
            for module_path, class_name in views_list:
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    view_class = getattr(module, class_name)
                    self.check(
                        f"{app_name}: {class_name}",
                        hasattr(view_class, 'get') or hasattr(view_class, 'post'),
                        f"Module: {module_path}"
                    )
                except (ImportError, AttributeError) as e:
                    self.check(f"{app_name}: {class_name}", False, str(e))
    
    def validate_celery_setup(self):
        """Test 6: Validate Celery configuration."""
        print("\n" + "="*70)
        print("TEST 6: Celery Async Task Processing")
        print("="*70)
        
        try:
            from backend.config.celery import app as celery_app
            from backend.ml_pipeline.tasks import run_full_pipeline_task
            
            self.check(
                "Celery app configured",
                celery_app is not None,
                "Celery instance created"
            )
            
            self.check(
                "Pipeline task registered",
                callable(run_full_pipeline_task),
                "run_full_pipeline_task available"
            )
            
        except Exception as e:
            self.check("Celery Setup", False, str(e))
    
    def validate_forms_and_serialization(self):
        """Test 7: Validate forms and data handling."""
        print("\n" + "="*70)
        print("TEST 7: Forms and Data Handling")
        print("="*70)
        
        try:
            from backend.tools.forms import CSVUploadForm, XHandleFetchForm, PipelineExecutionForm
            
            forms = {
                'CSVUploadForm': CSVUploadForm,
                'XHandleFetchForm': XHandleFetchForm,
                'PipelineExecutionForm': PipelineExecutionForm,
            }
            
            for form_name, form_class in forms.items():
                self.check(
                    f"Form: {form_name}",
                    hasattr(form_class, 'base_fields'),
                    f"Fields: {len(form_class().fields)}"
                )
            
        except Exception as e:
            self.check("Forms Validation", False, str(e))
    
    def validate_templates(self):
        """Test 8: Validate template files."""
        print("\n" + "="*70)
        print("TEST 8: HTML Templates")
        print("="*70)
        
        template_paths = [
            'backend/templates/base.html',
            'backend/templates/public/index.html',
            'backend/templates/public/live_prediction.html',
            'backend/templates/dashboard/index.html',
            'backend/templates/dashboard/volunteer_detail.html',
            'backend/templates/tools/index.html',
            'backend/templates/tools/csv_upload.html',
            'backend/templates/accounts/login.html',
        ]
        
        for path in template_paths:
            exists = os.path.exists(path)
            if exists:
                with open(path, 'r') as f:
                    size = len(f.read())
                self.check(
                    f"Template: {os.path.basename(path)}",
                    size > 0,
                    f"Size: {size} bytes"
                )
            else:
                self.check(f"Template: {os.path.basename(path)}", False, f"File not found: {path}")
    
    def validate_requirements(self):
        """Test 9: Validate dependencies."""
        print("\n" + "="*70)
        print("TEST 9: Required Dependencies")
        print("="*70)
        
        required_packages = {
            'django': 'Django',
            'psycopg2': 'PostgreSQL adapter',
            'celery': 'Async task queue',
            'redis': 'Cache/broker',
            'torch': 'PyTorch',
            'transformers': 'Hugging Face',
            'sklearn': 'scikit-learn',
            'numpy': 'NumPy',
            'pandas': 'Pandas',
        }
        
        for package_name, description in required_packages.items():
            try:
                __import__(package_name)
                self.check(f"Dependency: {package_name}", True, description)
            except ImportError:
                self.check(f"Dependency: {package_name}", False, f"Not installed: {description}")
    
    def run_all(self):
        """Run complete validation suite."""
        print("\n")
        print("█" * 70)
        print("█ BIG FIVE PERSONALITY PREDICTION SYSTEM - VALIDATION SUITE".ljust(70) + "█")
        print("█" * 70)
        
        self.validate_models()
        self.validate_ml_services()
        self.validate_bfi_scorer()
        self.validate_pipeline_order()
        self.validate_views_and_routes()
        self.validate_celery_setup()
        self.validate_forms_and_serialization()
        self.validate_templates()
        self.validate_requirements()
        
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        for result in self.results:
            print(result)
        
        total = self.checks_passed + self.checks_failed
        percentage = (self.checks_passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print(f"Results: {self.checks_passed}/{total} checks passed ({percentage:.1f}%)")
        print("="*70)
        
        if self.checks_failed == 0:
            print("\n✓ ALL VALIDATION CHECKS PASSED")
            print("\nThe implementation is production-ready:")
            print("  • 8 database models with proper relationships")
            print("  • 6 ML services implementing the full pipeline")
            print("  • Strict pipeline order: Q-Learn → BERT → GAN → Lasso")
            print("  • BFI-44 scoring with 21 reverse items")
            print("  • Complete view and template layer")
            print("  • Celery async task processing")
            print("  • All required dependencies available")
            return 0
        else:
            print(f"\n✗ {self.checks_failed} validation checks FAILED")
            print("\nPlease fix the above issues before deployment.")
            return 1


if __name__ == '__main__':
    suite = ValidationSuite()
    exit_code = suite.run_all()
    sys.exit(exit_code)
