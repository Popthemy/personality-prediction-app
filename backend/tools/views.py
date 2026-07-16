"""Tools views for data management and pipeline execution."""
import csv
import logging
import json
import re
from io import StringIO, BytesIO
from django.views.generic import TemplateView, FormView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, HttpResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.contrib import messages
from .forms import CSVUploadForm, XHandleFetchForm, PipelineExecutionForm
from backend.core.models import VOLUNTEER, BFI_SURVEY, POST
from backend.core.services.bfi_scorer import BFIScorer, score_bfi_survey
from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
from backend.ml_pipeline.tasks import run_full_pipeline_task

logger = logging.getLogger(__name__)


class ToolsView(LoginRequiredMixin, TemplateView):
    """Main tools hub showing import, fetch, and pipeline options."""
    template_name = 'tools/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['volunteers_count'] = VOLUNTEER.objects.filter(
            researcher=self.request.user
        ).count()
        context['recent_volunteers'] = VOLUNTEER.objects.filter(
            researcher=self.request.user
        ).order_by('-created_at')[:5]
        # volunteers id
        context['volunteer_ids'] = list(VOLUNTEER.objects.filter(
            researcher=self.request.user).values_list('id', flat=True))
        return context


class CSVUploadView(LoginRequiredMixin, FormView):
    """Handle CSV import from Google Forms (BFI-44 responses)."""
    form_class = CSVUploadForm
    template_name = 'tools/csv_upload.html'
    success_url = reverse_lazy('tools:index')

    def form_valid(self, form):
        print("$$$$$$$$$$$$$ CSV Upload form is valid")
        csv_file = form.cleaned_data['csv_file']
        try:
            # Read CSV
            stream = StringIO(csv_file.read().decode('utf-8'))
            reader = csv.DictReader(stream)
            print(f"CSV Headers: {reader.fieldnames}")

            processed_count = 0

            for row in reader:
                # Extract twitter handle and BFI responses
                twitter_handle = row.get(
                    'X / Twitter Profile handle (if you have one)', '').strip()
                informed_consent = row.get(
                    'Informed Consent', '').strip().lower() == 'yes'
                if not informed_consent:
                    continue

                if not twitter_handle:
                    continue

                # Create or get volunteer
                volunteer, created = VOLUNTEER.objects.get_or_create(
                    x_handle=twitter_handle,
                    consent_given=True if informed_consent else False,
                    defaults={'researcher': self.request.user,}
                )

                # Extract BFI-44 responses (questions 1-44)
                responses = {}
                for header, value in row.items():
                    print(f"Processing header: {header}, value: {value}")
                    match = re.match(r'^(\d+)\.', header.strip())
                    if not match:
                        continue
                    item = int(match.group(1))
                    if 1 <= item <= 44:
                        try:
                            responses[str(item)] = int(value)
                        except ValueError:
                            logger.warning(
                                f"Invalid response for item {item}: {value}")

                if len(responses) >= 40:  # At least 90% responses
                    # Calculate BFI scores
                    scores = score_bfi_survey(responses)

                    # Save BFI survey
                    bfi_survey =  BFI_SURVEY.objects.create(
                        volunteer=volunteer,
                        responses=responses,
                        openness=scores['Openness'],
                        conscientiousness=scores['Conscientiousness'],
                        extraversion=scores['Extraversion'],
                        agreeableness=scores['Agreeableness'],
                        neuroticism=scores['Neuroticism'],
                        completed_at=timezone.now(),
                    )
                    volunteer.bfi_surveys.add(bfi_survey)
                    processed_count += 1

            messages.success(
                self.request,
                f'Successfully processed {processed_count} volunteer records!'
            )
            logger.info(
                f"User {self.request.user} imported {processed_count} BFI surveys"
            )
        except Exception as e:
            logger.error(f"CSV import error: {str(e)}")
            messages.error(self.request, f'Error processing CSV: {str(e)}')
            return super().form_invalid(form)

        return super().form_valid(form)


class FetchPostsView(LoginRequiredMixin, FormView):
    """Fetch X (Twitter) posts for a volunteer."""
    form_class = XHandleFetchForm
    template_name = 'tools/fetch_posts.html'
    success_url = reverse_lazy('tools:index')

    def form_valid(self, form):
        x_handle = form.cleaned_data['x_handle']
        try:
            volunteer = VOLUNTEER.objects.get(
                x_handle=x_handle,
                researcher=self.request.user
            )

            from backend.core.services.twitter_fetcher import TwitterFetcher
            fetcher = TwitterFetcher()
            saved, skipped = fetcher.fetch_and_save(volunteer)

            if saved > 0:
                messages.success(
                    self.request,
                    f'Fetched {saved} new posts for @{x_handle} ({skipped} already existed).'
                )
            elif skipped > 0:
                messages.info(
                    self.request,
                    f'No new posts found for @{x_handle} — {skipped} already in database.'
                )
            else:
                messages.warning(
                    self.request,
                    f'Could not fetch posts for @{x_handle}. The Nitter instance may be unavailable.'
                )

            logger.info(f"Posts fetched for @{x_handle}: saved={saved}, skipped={skipped}")

        except VOLUNTEER.DoesNotExist:
            messages.error(
                self.request,
                f'Volunteer @{x_handle} not found'
            )
        except Exception as e:
            logger.error(f"FetchPostsView error for @{x_handle}: {e}")
            messages.error(self.request, f'Error fetching posts: {str(e)}')

        return super().form_valid(form)


class RunPipelineView(LoginRequiredMixin, FormView):
    """Trigger the full ML pipeline for a volunteer."""
    form_class = PipelineExecutionForm
    template_name = 'tools/run_pipeline.html'
    success_url = reverse_lazy('dashboard:index')

    def form_valid(self, form):
        volunteer_id = form.cleaned_data['volunteer_id']
        try:
            volunteer = VOLUNTEER.objects.get(
                id=volunteer_id,
                researcher=self.request.user
            )

            # Check prerequisites
            if not BFI_SURVEY.objects.filter(volunteer=volunteer).exists():
                messages.warning(
                    self.request,
                    'Please import BFI-44 ground truth first'
                )
                return redirect('tools:csv_upload')

            # Queue the full pipeline task
            run_full_pipeline_task.delay(volunteer_id)

            messages.success(
                self.request,
                f'Pipeline started for @{volunteer.x_handle}. Check status in dashboard.'
            )
            logger.info(
                f"Pipeline triggered for volunteer {volunteer_id} by {self.request.user}"
            )
        except VOLUNTEER.DoesNotExist:
            messages.error(self.request, 'Volunteer not found')
        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            messages.error(self.request, f'Error: {str(e)}')

        return super().form_valid(form)


class AnalyzeProfileView(LoginRequiredMixin, FormView):
    """
    Manages fetching X posts and running predictions for any X handle,
    including those without ground truth BFI data.
    """
    form_class = XHandleFetchForm
    template_name = 'tools/analyze.html'

    def form_valid(self, form):
        x_handle = form.cleaned_data['x_handle']
        limit = form.cleaned_data.get('limit', 50)
        exclude_retweets = form.cleaned_data.get('exclude_retweets', True)

        try:
            from backend.core.models import BFI_SURVEY
            
            # Check if this handle has a volunteer and BFI survey
            volunteer_exists = VOLUNTEER.objects.filter(x_handle=x_handle).exists()
            has_bfi = False
            volunteer = None
            if volunteer_exists:
                volunteer = VOLUNTEER.objects.get(x_handle=x_handle)
                has_bfi = BFI_SURVEY.objects.filter(volunteer=volunteer).exists()

            is_survey_submission = self.request.POST.get('is_survey_submission') == 'true'

            if not has_bfi and not is_survey_submission:
                # First step: redirect to BFI Survey display
                return self.render_to_response(self.get_context_data(
                    form=form,
                    show_survey=True,
                    x_handle=x_handle,
                    limit=limit,
                    exclude_retweets=exclude_retweets
                ))

            if is_survey_submission:
                # Second step: Process BFI survey responses
                responses = {}
                for i in range(1, 45):
                    val = self.request.POST.get(f'q{i}')
                    if not val or not val.isdigit() or not (1 <= int(val) <= 5):
                        messages.error(self.request, f"Please answer all 44 questions. (Question {i} needs a rating).")
                        return self.render_to_response(self.get_context_data(
                            form=form,
                            show_survey=True,
                            x_handle=x_handle,
                            limit=limit,
                            exclude_retweets=exclude_retweets,
                            survey_responses=self.request.POST
                        ))
                    responses[str(i)] = int(val)

                from backend.core.services.bfi_scorer import BFIScorer
                traits = BFIScorer.calculate_scores(responses)

                # Ensure volunteer exists
                volunteer, created = VOLUNTEER.objects.get_or_create(
                    x_handle=x_handle,
                    defaults={
                        'researcher': self.request.user,
                        'consent_given': True,
                        'pipeline_status': 'idle'
                    }
                )

                # Create/update BFI Survey
                bfi_survey, _ = BFI_SURVEY.objects.update_or_create(
                    volunteer=volunteer,
                    defaults={
                        'responses': responses,
                        'openness': traits['openness'],
                        'conscientiousness': traits['conscientiousness'],
                        'extraversion': traits['extraversion'],
                        'agreeableness': traits['agreeableness'],
                        'neuroticism': traits['neuroticism'],
                    }
                )
                if not volunteer.bfi_surveys.filter(id=bfi_survey.id).exists():
                    volunteer.bfi_surveys.add(bfi_survey)

            # Ensure we have volunteer object instantiated at this stage
            if not volunteer:
                volunteer, created = VOLUNTEER.objects.get_or_create(
                    x_handle=x_handle,
                    defaults={
                        'researcher': self.request.user,
                        'consent_given': True,
                        'pipeline_status': 'idle'
                    }
                )

            # 2. Fetch posts
            from backend.core.services.twitter_fetcher import TwitterFetcher
            fetcher = TwitterFetcher()
            fetcher._max_posts = limit
            saved, skipped = fetcher.fetch_and_save(volunteer)



            posts_qs = POST.objects.filter(volunteer=volunteer)
            if exclude_retweets:
                posts_qs = posts_qs.filter(is_retweet=False)

            from backend.ml_pipeline.processors.text_preprocessor import TextPreprocessor
            preprocessor = TextPreprocessor()

            valid_posts = []
            for post in posts_qs:
                cleaned = preprocessor.clean(post.content)
                if preprocessor.is_valid(cleaned):
                    post.cleaned_content = cleaned
                    valid_posts.append(post)

            if len(valid_posts) == 0:
                messages.error(
                    self.request,
                    f"No valid posts found for @{x_handle} to analyze. Please try a different handle."
                )
                return self.form_invalid(form)

            # 4. Q-Learning Selection
            from backend.ml_pipeline.services.qlearning_agent import QLearningAgent, create_post_features
            agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.05)

            post_features = []
            for post in valid_posts:
                features = create_post_features(post)
                post_features.append({
                    'id': post.id,
                    'content': post.cleaned_content,
                    **features,
                })

            top_k = min(10, len(valid_posts))
            selected = agent.select_posts(post_features, top_k=top_k, training=False)
            selected_post_ids = {s['id'] for s in selected}

            selected_posts = []
            for post in valid_posts:
                if post.id in selected_post_ids:
                    post.selected_by_qlearning = True
                    for s in selected:
                        if s['id'] == post.id:
                            post.q_value = s.get('q_value', 0)
                            break
                    selected_posts.append(post)
                else:
                    post.selected_by_qlearning = False
                post.save()

            # 5. Extract BERT Embeddings
            from backend.ml_pipeline.services.bert_encoder import BERTEncoder
            import time
            encoder = BERTEncoder()

            embeddings = []
            from backend.core.models import BERT_EMBEDDING, PSYCHOMETRIC_PROFILE, LASSO_MODEL

            for post in selected_posts:
                start_time = time.time()
                # Check idempotency: does BERT_EMBEDDING already exist for this post?
                emb_obj = BERT_EMBEDDING.objects.filter(post=post).first()
                if not emb_obj:
                    result = encoder.encode_text(post.cleaned_content)
                    emb_obj = BERT_EMBEDDING.objects.create(
                        post=post,
                        volunteer=volunteer,
                        embedding_vector=result['embedding'],
                        model_name=result['model_name'],
                        processing_time_seconds=time.time() - start_time,
                    )
                embeddings.append(emb_obj)
                post.embedding_processed = True
                post.save()

            # 6. Check BFI survey ground truth to decide prediction workflow
            from backend.core.models import BFI_SURVEY
            has_bfi = BFI_SURVEY.objects.filter(volunteer=volunteer).exists()

            if has_bfi:
                # If ground truth exists, run the full training + prediction orchestrator pipeline
                orchestrator = PipelineOrchestrator(volunteer.id)
                orchestrator.run_full_pipeline()
                messages.success(
                    self.request,
                    f"Successfully ran training pipeline and predicted personality for @{x_handle}."
                )
            else:
                # Predict using existing trained LASSO_MODEL coefficients from database
                import numpy as np
                X_pred = []
                for emb in embeddings:
                    if isinstance(emb.embedding_vector, str):
                        embedding_list = json.loads(emb.embedding_vector)
                    else:
                        embedding_list = emb.embedding_vector
                    X_pred.append(embedding_list)
                X_pred = np.array(X_pred)

                # Normalize features on user's own selection
                X_norm = (X_pred - X_pred.mean(axis=0)) / (X_pred.std(axis=0) + 1e-8)

                predictions = {}
                traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

                for t in traits:
                    model_db = LASSO_MODEL.objects.filter(trait=t).order_by('-id').first()

                    if not model_db:
                        # Cold start: bootstrap training on the first volunteer possessing a ground truth
                        first_gt_vol = VOLUNTEER.objects.filter(bfi_surveys__isnull=False).first()
                        if first_gt_vol:
                            PipelineOrchestrator(first_gt_vol.id).run_full_pipeline()
                            model_db = LASSO_MODEL.objects.filter(trait=t).order_by('-id').first()

                    if model_db:
                        coefs_dict = json.loads(model_db.coefficients)
                        w = np.array([coefs_dict[str(i)] for i in range(768)])
                        b = model_db.intercept

                        preds_norm = np.dot(X_norm, w) + b
                        preds_norm = np.clip(preds_norm, 0, 1)
                        preds = preds_norm * 4.0 + 1.0  # Denormalize
                        predictions[f'predicted_{t}'] = float(np.mean(preds))
                    else:
                        # Clear heuristic default
                        predictions[f'predicted_{t}'] = 3.0

                # Save predictions to PSYCHOMETRIC_PROFILE
                profile, _ = PSYCHOMETRIC_PROFILE.objects.update_or_create(
                    volunteer=volunteer,
                    defaults={
                        'predicted_openness': predictions['predicted_openness'],
                        'predicted_conscientiousness': predictions['predicted_conscientiousness'],
                        'predicted_extraversion': predictions['predicted_extraversion'],
                        'predicted_agreeableness': predictions['predicted_agreeableness'],
                        'predicted_neuroticism': predictions['predicted_neuroticism'],
                        'overall_mae': None,
                        'posts_analyzed': len(selected_posts),
                        'embeddings_used': len(embeddings),
                        'synthetic_data_used': 0,
                        'prediction_confidence': 0.85,
                        'personality_summary': f"Personality profile predicted from X posts using Lasso regression.",
                    }
                )

                messages.success(
                    self.request,
                    f"Successfully fetched posts and predicted personality for @{x_handle} using trained model."
                )

            volunteer.pipeline_status = 'success'
            volunteer.save()
            return redirect('dashboard:volunteer_detail', pk=volunteer.id)

        except Exception as e:
            logger.error(f"AnalyzeProfileView error for @{x_handle}: {e}", exc_info=True)
            messages.error(self.request, f"Error analyzing profile: {str(e)}")
            return self.form_invalid(form)

