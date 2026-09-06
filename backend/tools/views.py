"""Tools views for data management and pipeline execution."""
import csv
import logging
import json
import re
from pathlib import Path
from io import StringIO, BytesIO
from django.views.generic import TemplateView, FormView, View, DetailView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, HttpResponse, FileResponse, Http404
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.contrib import messages
from .forms import (
    CSVUploadForm,
    XHandleFetchForm,
    PipelineExecutionForm,
    PipelineControlForm,
    CohortTrainingForm,
    PredictionSelectionForm,
)
from backend.core.models import (
    VOLUNTEER, BFI_SURVEY, POST, BERT_EMBEDDING, COHORT_MODEL, PSYCHOMETRIC_PROFILE,
    PANDORA_EXPERIMENT_RUN, PANDORA_CONDITION_RESULT, PANDORA_THRESHOLD_RESULT,
    PANDORA_DATASET_ALLOCATION,
)
from backend.core.services.bfi_scorer import BFIScorer, score_bfi_survey
from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
from backend.ml_pipeline.services.timeline_exporter import export_cleaned_posts_to_txt
from backend.ml_pipeline.tasks import run_full_pipeline_task, run_pipeline_phase_task

logger = logging.getLogger(__name__)

PANDORA_DATA_DIR = Path("PANDORA") / "pandora-big5" / "data"


def _pandora_dataset_status():
    data_dir = PANDORA_DATA_DIR
    parquet_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    prepared_cache = Path("pandora_personality") / "data" / "pandora_prepared.json"
    latest_run = PANDORA_EXPERIMENT_RUN.objects.order_by('-created_at').first()
    return {
        'path': str(data_dir.resolve()),
        'exists': data_dir.exists(),
        'parquet_count': len(parquet_files),
        'parquet_files': [p.name for p in parquet_files],
        'prepared_cache_exists': prepared_cache.exists(),
        'prepared_cache_path': str(prepared_cache.resolve()),
        'latest_run': latest_run,
    }


def _json_safe_float(value):
    try:
        numeric = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if numeric is None or numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _json_safe_value(value):
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    return value


def _record_pandora_bundle(run, bundle):
    sample = bundle.get("sample") or {}
    for user_id in sample.get("user_ids") or []:
        PANDORA_DATASET_ALLOCATION.objects.get_or_create(
            run=run,
            pandora_user_id=str(user_id),
            defaults={'source_split': 'experiment_sample', 'row_hash': str(user_id)},
        )

    comparison = bundle.get("comparison")
    if comparison is not None:
        for row in comparison.to_dict("records"):
            overall = (bundle.get("results") or {}).get(row.get("condition"), {}).get("overall", {})
            PANDORA_CONDITION_RESULT.objects.update_or_create(
                run=run,
                condition=row.get("condition", ""),
                defaults={
                    'description': row.get("description") or row.get("label") or "",
                    'selection': row.get("selection") or "",
                    'gan': bool(row.get("gan")),
                    'model': row.get("model") or "",
                    'val_mae': _json_safe_float(row.get("val_mae")),
                    'accuracy': _json_safe_float(row.get("accuracy")),
                    'macro_f1': _json_safe_float(row.get("macro_f1")),
                    'specificity': _json_safe_float(overall.get("specificity")),
                    'precision': _json_safe_float(overall.get("macro_precision")),
                    'recall': _json_safe_float(overall.get("macro_recall")),
                    'metrics': _json_safe_value(row),
                }
            )

    sweeps = bundle.get("threshold_sweeps")
    if sweeps is not None:
        PANDORA_THRESHOLD_RESULT.objects.filter(run=run).delete()
        for row in sweeps.to_dict("records"):
            threshold = _json_safe_float(row.get("threshold"))
            if threshold is None:
                continue
            PANDORA_THRESHOLD_RESULT.objects.create(
                run=run,
                condition=row.get("condition", ""),
                split=row.get("split") or "validation",
                trait=row.get("trait") or "",
                threshold=threshold,
                accuracy=_json_safe_float(row.get("accuracy")),
                f1_score=_json_safe_float(row.get("f1_score")),
                specificity=_json_safe_float(row.get("specificity")),
                precision=_json_safe_float(row.get("precision")),
                recall=_json_safe_float(row.get("recall")),
            )

    findings = bundle.get("findings") or {}
    best = findings.get("best_condition") or {}
    run.artifact_dir = str(bundle.get("artifact_dir") or "")
    run.best_condition = best.get("condition") or ""
    run.best_accuracy = _json_safe_float(best.get("accuracy"))
    run.best_f1 = _json_safe_float(best.get("macro_f1"))
    run.audit_status = (bundle.get("audit") or {}).get("status", "")
    run.findings = findings.get("notes", [])
    run.summary = _json_safe_value({
        'comparison': comparison.to_dict("records") if comparison is not None else [],
        'artifact_dir': run.artifact_dir,
    })
    run.status = 'completed'
    run.completed_at = timezone.now()
    run.save()


def _export_cleaned_timeline_for_volunteer(volunteer, posts):
    """Clean timeline posts and export them to a handle-named text file."""
    from backend.ml_pipeline.processors.text_preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    cleaned_posts = []
    for post in posts:
        cleaned = preprocessor.clean(post.content)
        if preprocessor.is_valid(cleaned):
            cleaned_posts.append(cleaned)

    export_path = export_cleaned_posts_to_txt(volunteer.x_handle, cleaned_posts)
    logger.info(
        "Exported cleaned timeline for @%s to %s (%s cleaned posts)",
        volunteer.x_handle,
        export_path,
        len(cleaned_posts),
    )
    return export_path, len(cleaned_posts)


class ToolsView(LoginRequiredMixin, TemplateView):
    """Main tools hub for PANDORA experiments and prediction evaluation."""
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
        context['pipeline_control_form'] = PipelineControlForm(user=self.request.user)
        context['active_cohort_model'] = COHORT_MODEL.objects.filter(is_active=True).order_by('-updated_at').first()
        context['pandora_status'] = _pandora_dataset_status()
        context['latest_pandora_runs'] = PANDORA_EXPERIMENT_RUN.objects.filter(
            researcher=self.request.user
        ).order_by('-created_at')[:5]
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
                    # Calculate BFI scores (lowercase keys from calculate_scores)
                    scores = score_bfi_survey(responses)

                    # Upsert BFI survey — handles re-uploads without crashing
                    BFI_SURVEY.objects.update_or_create(
                        volunteer=volunteer,
                        defaults={
                            'responses': responses,
                            'openness': scores.get('Openness') or scores.get('openness'),
                            'conscientiousness': scores.get('Conscientiousness') or scores.get('conscientiousness'),
                            'extraversion': scores.get('Extraversion') or scores.get('extraversion'),
                            'agreeableness': scores.get('Agreeableness') or scores.get('agreeableness'),
                            'neuroticism': scores.get('Neuroticism') or scores.get('neuroticism'),
                            'completed_at': timezone.now(),
                        }
                    )
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
            posts = POST.objects.filter(volunteer=volunteer).order_by('-created_at_original')
            export_path, cleaned_count = _export_cleaned_timeline_for_volunteer(volunteer, posts)

            if saved > 0:
                messages.success(
                    self.request,
                    f'Fetched {saved} new posts for @{x_handle} ({skipped} already existed). '
                    f'Cleaned timeline exported to {export_path.name} ({cleaned_count} posts).'
                )
            elif skipped > 0:
                messages.info(
                    self.request,
                    f'No new posts found for @{x_handle} — {skipped} already in database. '
                    f'Cleaned timeline exported to {export_path.name} ({cleaned_count} posts).'
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

            volunteer.pipeline_status = 'processing'
            volunteer.save(update_fields=['pipeline_status'])

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


class PipelineControlView(LoginRequiredMixin, FormView):
    """Unified control surface for full pipeline and manual phase execution."""
    form_class = PipelineControlForm
    success_url = reverse_lazy('tools:index')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def form_valid(self, form):
        volunteer_id = form.cleaned_data['volunteer_id']
        action = form.cleaned_data['action']

        try:
            volunteer = VOLUNTEER.objects.get(
                id=volunteer_id,
                researcher=self.request.user
            )

            if action == 'full':
                if not BFI_SURVEY.objects.filter(volunteer=volunteer).exists():
                    messages.warning(
                        self.request,
                        'Please import BFI-44 ground truth first'
                    )
                    return redirect('tools:csv_upload')

                volunteer.pipeline_status = 'processing'
                volunteer.save(update_fields=['pipeline_status'])
                run_full_pipeline_task.delay(volunteer.id)
                messages.success(
                    self.request,
                    f'Full pipeline queued for @{volunteer.x_handle}.'
                )
            else:
                action_labels = {
                    'qlearning': 'Q-Learning',
                    'bert': 'BERT',
                    'gan': 'GAN',
                    'lasso': 'Lasso',
                }

                if action == 'qlearning':
                    if not POST.objects.filter(volunteer=volunteer).exists():
                        messages.warning(self.request, 'No posts found for Q-Learning.')
                        return redirect('tools:index')
                elif action == 'bert':
                    if not POST.objects.filter(volunteer=volunteer, selected_by_qlearning=True).exists():
                        messages.warning(self.request, 'Run Q-Learning before BERT.')
                        return redirect('tools:index')
                elif action == 'gan':
                    if not BERT_EMBEDDING.objects.filter(post__volunteer=volunteer).exists():
                        messages.warning(self.request, 'Run BERT before GAN augmentation.')
                        return redirect('tools:index')
                elif action == 'lasso':
                    if not BFI_SURVEY.objects.filter(volunteer=volunteer).exists():
                        messages.warning(self.request, 'Import BFI-44 ground truth before Lasso training.')
                        return redirect('tools:index')

                volunteer.pipeline_status = 'processing'
                volunteer.save(update_fields=['pipeline_status'])
                run_pipeline_phase_task.delay(volunteer.id, action)
                messages.success(
                    self.request,
                    f"{action_labels.get(action, action.title())} phase queued for @{volunteer.x_handle}."
                )

            logger.info(
                f"Pipeline control action={action} queued for volunteer {volunteer_id} by {self.request.user}"
            )
        except VOLUNTEER.DoesNotExist:
            messages.error(self.request, 'Volunteer not found')
        except Exception as e:
            logger.error(f"Pipeline control error: {str(e)}")
            messages.error(self.request, f'Error: {str(e)}')

        return super().form_valid(form)


class CohortTrainingView(LoginRequiredMixin, FormView):
    """Dedicated page for running the PANDORA 8-condition experiment."""
    form_class = CohortTrainingForm
    template_name = 'tools/train.html'
    success_url = reverse_lazy('tools:train')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['pandora_status'] = _pandora_dataset_status()
        context['runs'] = PANDORA_EXPERIMENT_RUN.objects.filter(
            researcher=self.request.user
        ).order_by('-created_at')[:8]
        context['consumed_sample_count'] = PANDORA_DATASET_ALLOCATION.objects.values(
            'pandora_user_id'
        ).distinct().count()

        return context

    def form_valid(self, form):
        status = _pandora_dataset_status()
        if not status['exists'] or status['parquet_count'] == 0:
            messages.error(self.request, 'PANDORA parquet files were not found in the project dataset folder.')
            return self.form_invalid(form)

        sample_size = form.cleaned_data['sample_size']
        seed = form.cleaned_data['seed']
        run = PANDORA_EXPERIMENT_RUN.objects.create(
            researcher=self.request.user,
            run_id=f"pandora_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
            label=form.cleaned_data.get('run_label') or "",
            status='running',
            dataset_path=status['path'],
            sample_size=sample_size,
            seed=seed,
            max_comments_per_user=form.cleaned_data.get('max_comments_per_user'),
            refresh_prepared_cache=form.cleaned_data.get('refresh_prepared_cache') or False,
            allow_reuse=form.cleaned_data.get('allow_reuse') or False,
            started_at=timezone.now(),
        )

        try:
            from backend.ml_pipeline.experiments import pandora_runner

            pandora_file = pandora_runner._default_pandora_file()
            if pandora_file is None:
                raise ValueError("No PANDORA parquet file found.")

            work_dir = Path("pandora_personality")
            data_dir = work_dir / "data"
            cache_dir = work_dir / "cache"
            artifact_dir = work_dir / "artifacts"
            for folder in (data_dir, cache_dir, artifact_dir):
                folder.mkdir(parents=True, exist_ok=True)

            prepared = pandora_runner.load_or_prepare_pandora(
                pandora_file,
                data_dir / "pandora_prepared.json",
                refresh_prepared=form.cleaned_data.get('refresh_prepared_cache') or False,
            )
            if not run.allow_reuse:
                used_ids = set(PANDORA_DATASET_ALLOCATION.objects.values_list('pandora_user_id', flat=True))
                prepared = [item for item in prepared if item.user_id not in used_ids]
                if len(prepared) < sample_size:
                    raise ValueError(
                        f"Only {len(prepared)} unused PANDORA users remain. Reduce sample size or enable reuse."
                    )

            cfg = pandora_runner.ExperimentConfig(
                sample_n_users=sample_size,
                seed=seed,
                embedding_cache_dir=str(cache_dir),
                output_dir=str(artifact_dir),
            )
            bundle = pandora_runner.ExperimentRunner(prepared, cfg).run()
            _record_pandora_bundle(run, bundle)

            messages.success(
                self.request,
                f"PANDORA experiment completed. Artifacts saved to {run.artifact_dir}."
            )
            return redirect('tools:experiment_detail', pk=run.pk)
        except Exception as e:
            run.status = 'error'
            run.error_message = str(e)
            run.completed_at = timezone.now()
            run.save(update_fields=['status', 'error_message', 'completed_at', 'updated_at'])
            logger.error("PANDORA experiment error: %s", e, exc_info=True)
            messages.error(self.request, f'Error running PANDORA experiment: {str(e)}')
            return self.form_invalid(form)

        return super().form_valid(form)


class PandoraRunListView(LoginRequiredMixin, ListView):
    """History of PANDORA experiment runs."""
    model = PANDORA_EXPERIMENT_RUN
    template_name = 'tools/experiment_history.html'
    context_object_name = 'runs'
    paginate_by = 20

    def get_queryset(self):
        return PANDORA_EXPERIMENT_RUN.objects.filter(researcher=self.request.user).order_by('-created_at')


class PandoraRunDetailView(LoginRequiredMixin, DetailView):
    """Presentation-ready PANDORA run results."""
    model = PANDORA_EXPERIMENT_RUN
    template_name = 'tools/experiment_detail.html'
    context_object_name = 'run'

    def get_queryset(self):
        return PANDORA_EXPERIMENT_RUN.objects.filter(researcher=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        run = self.object
        context['conditions'] = run.condition_results.all()
        context['thresholds'] = run.threshold_results.all()
        context['plot_files'] = []
        if run.artifact_dir:
            plot_dir = Path(run.artifact_dir) / "plots"
            if plot_dir.exists():
                context['plot_files'] = [
                    {'name': p.name, 'url': reverse_lazy('tools:experiment_plot', kwargs={'pk': run.pk, 'filename': p.name})}
                    for p in sorted(plot_dir.glob("*.png"))
                ]
        return context


class PandoraRunPlotView(LoginRequiredMixin, View):
    """Serve generated plot images for a user's own PANDORA run."""

    def get(self, request, pk, filename):
        run = PANDORA_EXPERIMENT_RUN.objects.filter(pk=pk, researcher=request.user).first()
        if not run or not run.artifact_dir:
            raise Http404("Plot not found")
        plot_dir = (Path(run.artifact_dir) / "plots").resolve()
        target = (plot_dir / Path(filename).name).resolve()
        if plot_dir not in target.parents or not target.exists() or target.suffix.lower() != ".png":
            raise Http404("Plot not found")
        return FileResponse(target.open("rb"), content_type="image/png")


class CohortPredictionView(LoginRequiredMixin, FormView):
    """Prediction-only page that uses the saved cohort model artifact."""
    form_class = PredictionSelectionForm
    template_name = 'tools/predict.html'
    success_url = reverse_lazy('tools:predict')

    def _get_prediction_volunteers(self):
        active_model = COHORT_MODEL.objects.filter(is_active=True).order_by('-updated_at').first()
        if not active_model:
            return VOLUNTEER.objects.none()

        return VOLUNTEER.objects.filter(
            researcher=self.request.user,
            id__in=active_model.validation_volunteer_ids,
        ).order_by('x_handle')

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['volunteers'] = self._get_prediction_volunteers()
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        active_model = COHORT_MODEL.objects.filter(is_active=True).order_by('-updated_at').first()
        prediction_volunteers = self._get_prediction_volunteers()

        context['active_model'] = active_model
        context['prediction_volunteers'] = prediction_volunteers
        context['prediction_result'] = kwargs.get('prediction_result')
        context['prediction_profile'] = kwargs.get('prediction_profile')
        context['available_count'] = prediction_volunteers.count()
        return context

    def form_valid(self, form):
        volunteer_id = form.cleaned_data['volunteer_id']
        try:
            volunteer = VOLUNTEER.objects.get(
                id=volunteer_id,
                researcher=self.request.user,
            )

            active_model = COHORT_MODEL.objects.filter(is_active=True).order_by('-updated_at').first()
            if not active_model:
                messages.warning(self.request, 'Train a cohort model before running prediction.')
                return self.form_invalid(form)

            if volunteer_id not in set(active_model.validation_volunteer_ids):
                messages.warning(self.request, 'Select a volunteer from the held-out prediction split.')
                return self.form_invalid(form)

            orchestrator = PipelineOrchestrator(volunteer.id)
            result = orchestrator.predict_from_saved_model()
            prediction_result = result['prediction_result']

            pipeline_summary = {
                'mode': 'prediction_only',
                'model_id': result.get('model_id'),
                'model_version': result.get('model_version'),
                'validation_handles': result.get('validation_handles', []),
                'train_handles': result.get('train_handles', []),
            }
            orchestrator._save_psychometric_profile(prediction_result, pipeline_summary=pipeline_summary)
            profile = PSYCHOMETRIC_PROFILE.objects.filter(volunteer=volunteer).first()

            messages.success(
                self.request,
                f"Prediction completed for @{volunteer.x_handle} using the saved cohort model."
            )
            logger.info(
                "Prediction-only run completed for @%s by %s with model %s",
                volunteer.x_handle,
                self.request.user,
                result.get('model_version'),
            )
            return self.render_to_response(
                self.get_context_data(
                    form=form,
                    prediction_result=prediction_result,
                    prediction_profile=profile,
                )
            )
        except VOLUNTEER.DoesNotExist:
            messages.error(self.request, 'Volunteer not found')
        except Exception as e:
            logger.error("Prediction-only error: %s", e, exc_info=True)
            messages.error(self.request, f'Error running prediction: {str(e)}')
        return self.form_invalid(form)


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
        volunteer = None

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
                        'pipeline_status': 'pending'
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

            # Ensure we have volunteer object instantiated at this stage
            if not volunteer:
                volunteer, created = VOLUNTEER.objects.get_or_create(
                    x_handle=x_handle,
                defaults={
                        'researcher': self.request.user,
                        'consent_given': True,
                        'pipeline_status': 'pending'
                    }
                )

            # 2. Fetch posts
            volunteer.pipeline_status = 'processing'
            volunteer.save(update_fields=['pipeline_status'])

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

            export_path = export_cleaned_posts_to_txt(
                volunteer.x_handle,
                [post.cleaned_content for post in valid_posts],
            )

            if len(valid_posts) == 0:
                messages.error(
                    self.request,
                    f"No valid posts found for @{x_handle} to analyze. Please try a different handle. "
                    f"Cleaned timeline exported to {export_path.name}."
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

                predictions = {}
                traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

                for t in traits:
                    model_db = LASSO_MODEL.objects.filter(trait=t).order_by('-id').first()

                    if not model_db:
                        # Cold start: bootstrap training on the first volunteer possessing a ground truth
                        first_gt_vol = VOLUNTEER.objects.filter(bfi_survey__isnull=False).first()
                        if first_gt_vol:
                            PipelineOrchestrator(first_gt_vol.id).run_full_pipeline()
                            model_db = LASSO_MODEL.objects.filter(trait=t).order_by('-id').first()

                    if model_db:
                        coefs_dict = json.loads(model_db.coefficients)
                        feature_meta = coefs_dict.pop('_feature_metadata', {})
                        w = np.array([coefs_dict[str(i)] for i in range(768)])
                        b = model_db.intercept

                        feature_mean = feature_meta.get('feature_mean')
                        feature_scale = feature_meta.get('feature_scale')
                        if feature_mean and feature_scale:
                            feature_mean = np.array(feature_mean)
                            feature_scale = np.array(feature_scale)
                            X_norm = (X_pred - feature_mean) / (feature_scale + 1e-8)
                        else:
                            X_norm = (X_pred - X_pred.mean(axis=0)) / (X_pred.std(axis=0) + 1e-8)

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
                        'prediction_confidence': round(
                            min(
                                0.95,
                                max(
                                    0.55,
                                    0.60
                                    + min(len(selected_posts) / 80.0, 0.18)
                                    + min(len(embeddings) / 120.0, 0.12),
                                ),
                            ),
                            2,
                        ),
                        'personality_summary': f"Personality profile predicted from X posts using Lasso regression.",
                    }
                )

                messages.success(
                    self.request,
                    f"Successfully fetched posts and predicted personality for @{x_handle} using trained model."
                )

            volunteer.pipeline_status = 'completed'
            volunteer.save(update_fields=['pipeline_status'])
            return redirect('dashboard:volunteer_detail', pk=volunteer.id)

        except Exception as e:
            logger.error(f"AnalyzeProfileView error for @{x_handle}: {e}", exc_info=True)
            if volunteer:
                volunteer.pipeline_status = 'error'
                volunteer.save(update_fields=['pipeline_status'])
            messages.error(self.request, f"Error analyzing profile: {str(e)}")
            return self.form_invalid(form)

