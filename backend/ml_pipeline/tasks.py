"""Celery tasks for async ML pipeline execution."""
import logging
from celery import shared_task
from django.utils import timezone
from backend.core.models import VOLUNTEER, TRAINING_JOB

logger = logging.getLogger(__name__)


def _update_training_job(job: TRAINING_JOB, *, status=None, stage=None, progress=None, message=None, error=None, result=None):
    if status is not None:
        job.status = status
    if stage is not None:
        job.stage = stage
    if progress is not None:
        job.progress = int(progress)
    if message is not None:
        job.message = message
    if error is not None:
        job.error = error
    if result is not None:
        job.result = result
    if job.started_at is None and job.status == 'running':
        job.started_at = timezone.now()
    if job.status in {'completed', 'failed'} and job.finished_at is None:
        job.finished_at = timezone.now()
    job.save(update_fields=[
        'status', 'stage', 'progress', 'message', 'error', 'result',
        'started_at', 'finished_at', 'updated_at'
    ])


@shared_task(bind=True, max_retries=3)
def run_full_pipeline_task(self, volunteer_id):
    """
    Execute the complete ML pipeline for a volunteer.

    Pipeline order (STRICT):
    1. Q-Learning Active Signal Selection
    2. BERT Contextual Embedding Extraction
    3. GAN Data Augmentation (training only)
    4. Lasso Regression → Final Prediction

    Delegates entirely to PipelineOrchestrator which implements
    the correct, schema-compliant pipeline logic.
    """
    try:
        volunteer = VOLUNTEER.objects.get(id=volunteer_id)
        logger.info(f"Starting pipeline task for volunteer @{volunteer.x_handle}")

        from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(volunteer_id)
        result = orchestrator.run_full_pipeline()

        if result['status'] == 'error':
            raise Exception(result.get('error', 'Unknown pipeline error'))

        logger.info(
            f"Pipeline completed for @{volunteer.x_handle}. "
            f"Steps: {result.get('steps_completed', [])}. "
            f"Overall MAE: {result.get('metrics', {}).get('overall_mae', 'N/A')}. "
            f"Synthetic saved: {result.get('metrics', {}).get('synthetic_samples_saved', 0)}. "
            f"Synthetic reused: {result.get('metrics', {}).get('synthetic_samples_reused', 0)}. "
            f"Training mode: {result.get('metrics', {}).get('training_mode', 'unknown')}"
        )
        return result

    except VOLUNTEER.DoesNotExist:
        logger.error(f"Volunteer {volunteer_id} not found")
        return {'status': 'error', 'error': f'Volunteer {volunteer_id} not found'}

    except Exception as e:
        logger.error(f"Pipeline error for volunteer {volunteer_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def run_pipeline_phase_task(self, volunteer_id, phase):
    """Run a single pipeline phase asynchronously."""
    try:
        from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(volunteer_id)
        phase_map = {
            'qlearning': orchestrator.run_qlearning_phase,
            'bert': orchestrator.run_bert_phase,
            'gan': orchestrator.run_gan_phase,
            'lasso': orchestrator.run_lasso_phase,
        }

        if phase == 'full':
            result = orchestrator.run_full_pipeline()
        elif phase in phase_map:
            result = phase_map[phase]()
        else:
            raise ValueError(f"Unknown pipeline phase: {phase}")

        if result.get('status') == 'error':
            raise Exception(result.get('error', 'Unknown pipeline phase error'))

        return result

    except Exception as e:
        logger.error(f"Pipeline phase error for volunteer {volunteer_id} phase {phase}: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def train_cohort_model_task(self, researcher_id, split_seed=42):
    """Train and persist the shared cohort model artifact."""
    try:
        from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator

        volunteer = VOLUNTEER.objects.filter(
            researcher_id=researcher_id,
            bfi_survey__isnull=False,
        ).order_by('id').first()
        if not volunteer:
            raise ValueError("No labeled volunteer found for cohort training")

        orchestrator = PipelineOrchestrator(volunteer.id)
        result = orchestrator.train_cohort_model(split_seed=split_seed)

        logger.info(
            "Cohort model trained for researcher %s: train=%s validation=%s model=%s",
            researcher_id,
            result.get('train_count', 0),
            result.get('validation_count', 0),
            result.get('model_id'),
        )
        return result

    except Exception as e:
        logger.error(f"Cohort training error for researcher {researcher_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=0)
def train_pandora_model_task(self, job_id, researcher_id, split_seed=42):
    """Train the PANDORA-backed shared cohort model while updating a job row."""
    job = None
    try:
        job = TRAINING_JOB.objects.get(id=job_id, researcher_id=researcher_id)
        researcher = job.researcher

        _update_training_job(
            job,
            status='running',
            stage='queued',
            progress=1,
            message='Starting PANDORA training job...',
        )

        from backend.ml_pipeline.services.pandora_trainer import PandoraTrainingService

        def callback(stage: str, progress: int, message: str):
            _update_training_job(
                job,
                status='running',
                stage=stage,
                progress=progress,
                message=message,
            )

        service = PandoraTrainingService(
            researcher,
            volunteer=None,
            split_seed=split_seed,
            use_gan=True,
            status_callback=callback,
        )
        result = service.train()

        _update_training_job(
            job,
            status='completed',
            stage='completed',
            progress=100,
            message='Training finished successfully.',
            result=result,
        )
        return result

    except Exception as e:
        logger.error("PANDORA training job %s failed: %s", job_id, e, exc_info=True)
        if job is not None:
            _update_training_job(
                job,
                status='failed',
                stage='failed',
                progress=100 if getattr(job, 'progress', 0) < 100 else job.progress,
                message='Training failed.',
                error=str(e),
            )
        raise


@shared_task(bind=True, max_retries=3)
def predict_with_saved_model_task(self, volunteer_id):
    """Run prediction only using the saved cohort model."""
    try:
        from backend.ml_pipeline.services.pipeline_orchestrator import PipelineOrchestrator
        from backend.core.models import PSYCHOMETRIC_PROFILE

        volunteer = VOLUNTEER.objects.get(id=volunteer_id)
        orchestrator = PipelineOrchestrator(volunteer_id)
        result = orchestrator.predict_from_saved_model()
        prediction_result = result['prediction_result']
        orchestrator._save_psychometric_profile(
            prediction_result,
            pipeline_summary={
                'mode': 'prediction_only',
                'model_id': result.get('model_id'),
                'model_version': result.get('model_version'),
                'validation_handles': result.get('validation_handles', []),
                'train_handles': result.get('train_handles', []),
            }
        )

        profile = PSYCHOMETRIC_PROFILE.objects.filter(volunteer=volunteer).first()
        logger.info(
            "Prediction completed for @%s with model %s",
            volunteer.x_handle,
            result.get('model_version'),
        )
        return {
            'status': 'success',
            'volunteer_id': volunteer_id,
            'profile_id': profile.id if profile else None,
            'result': result,
        }

    except Exception as e:
        logger.error(f"Prediction task error for volunteer {volunteer_id}: {str(e)}")
        raise self.retry(exc=e, countdown=60)


@shared_task
def process_csv_batch(csv_file_path, researcher_id):
    """Process a batch of volunteers from CSV file."""
    # TODO: Implement batch CSV processing
    logger.info(f"CSV batch processing queued: {csv_file_path} for researcher {researcher_id}")
    pass


# ─── Metric helpers (used by external callers) ───────────────────────────────

def calculate_mae(ground_truth: dict, predictions: dict) -> float:
    """Calculate Mean Absolute Error between ground truth and predictions."""
    errors = [
        abs(ground_truth[trait] - predictions[trait])
        for trait in ground_truth.keys()
        if trait in predictions and ground_truth[trait] is not None and predictions[trait] is not None
    ]
    return sum(errors) / len(errors) if errors else 0.0


def calculate_correlation(ground_truth: dict, predictions: dict) -> float:
    """Calculate Pearson correlation between ground truth and predictions."""
    from scipy.stats import pearsonr
    try:
        keys = sorted(set(ground_truth.keys()) & set(predictions.keys()))
        gt_values = [ground_truth[t] for t in keys if ground_truth[t] is not None]
        pred_values = [predictions[t] for t in keys if predictions[t] is not None]
        if len(gt_values) < 2:
            return 0.0
        corr, _ = pearsonr(gt_values, pred_values)
        return float(corr)
    except Exception:
        return 0.0
