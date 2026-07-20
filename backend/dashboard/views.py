"""Dashboard views with analytics and personality insights."""
import logging
from django.views.generic import TemplateView, DetailView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Avg, Count
from backend.core.models import (
    VOLUNTEER, PSYCHOMETRIC_PROFILE, BFI_SURVEY,
    LASSO_MODEL, Q_LEARNING_LOG
)
from backend.ml_pipeline.services.insight_engine import build_domain_insights

logger = logging.getLogger(__name__)


class DashboardView(LoginRequiredMixin, TemplateView):
    """Main researcher dashboard with volunteer management and summary stats."""
    template_name = 'dashboard/index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        volunteers = VOLUNTEER.objects.filter(researcher=self.request.user)

        context['total_volunteers'] = volunteers.count()
        context['volunteers_with_bfi'] = volunteers.filter(
            bfi_survey__isnull=False
        ).count()
        context['volunteers_with_bfi_ids'] = set(
            volunteers.filter(bfi_survey__isnull=False).values_list('id', flat=True)
        )
        context['volunteers_with_predictions'] = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer__researcher=self.request.user
        ).values('volunteer').distinct().count()

        # Average performance metrics
        profiles = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer__researcher=self.request.user
        )
        if profiles.exists():
            context['avg_mae'] = profiles.aggregate(
                avg_mae=Avg('overall_mae')
            )['avg_mae']

        # Recent volunteers
        context['recent_volunteers'] = volunteers.order_by('-created_at')[:10]

        # Pipeline summary
        context['q_learning_actions'] = Q_LEARNING_LOG.objects.filter(
            volunteer__researcher=self.request.user
        ).count()

        return context


class VolunteerDetailView(LoginRequiredMixin, DetailView):
    """Detailed view of a single volunteer with OCEAN profile and predictions."""
    model = VOLUNTEER
    template_name = 'dashboard/volunteer_detail.html'
    context_object_name = 'volunteer'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        volunteer = self.get_object()

        # Ground truth BFI
        try:
            bfi_survey = volunteer.bfi_survey
        except Exception:
            bfi_survey = None
        context['ground_truth'] = bfi_survey

        # Prediction profile
        profile = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer=volunteer
        ).first()
        context['prediction'] = profile
        context['pipeline_summary'] = getattr(profile, 'pipeline_summary', {}) if profile else {}
        context['domain_insights'] = build_domain_insights(
            profile=profile,
            ground_truth=bfi_survey,
        ) if profile else {}

        if profile and profile.pipeline_summary:
            context['synthetic_samples_saved'] = profile.pipeline_summary.get('synthetic_samples_saved', 0)
            context['synthetic_samples_reused'] = profile.pipeline_summary.get('synthetic_samples_reused', 0)
            context['training_mode'] = profile.pipeline_summary.get('training_mode', 'unknown')
        else:
            context['synthetic_samples_saved'] = 0
            context['synthetic_samples_reused'] = 0
            context['training_mode'] = 'unknown'

        # Lasso model feature importance
        if profile:
            lasso_model = LASSO_MODEL.objects.filter(
                volunteer=volunteer
            ).first()
            if lasso_model:
                context['lasso_model'] = lasso_model

        # Q-Learning decisions
        context['q_learning_decisions'] = Q_LEARNING_LOG.objects.filter(
            volunteer=volunteer
        ).order_by('-created_at')[:5]

        return context


class DomainInsightsView(LoginRequiredMixin, TemplateView):
    """Aggregated insights across Education, Health, Employment, and Responsible AI domains."""
    template_name = 'dashboard/insights.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profiles = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer__researcher=self.request.user
        )

        if profiles.exists():
            # Domain-wise aggregation — using correct PSYCHOMETRIC_PROFILE field names
            context['education_insights'] = {
                'avg_conscientiousness': profiles.aggregate(
                    avg=Avg('predicted_conscientiousness')
                )['avg'],
                'count': profiles.count(),
            }
            context['health_insights'] = {
                'avg_neuroticism': profiles.aggregate(
                    avg=Avg('predicted_neuroticism')
                )['avg'],
                'avg_openness': profiles.aggregate(
                    avg=Avg('predicted_openness')
                )['avg'],
                'count': profiles.count(),
            }
            context['employment_insights'] = {
                'avg_extraversion': profiles.aggregate(
                    avg=Avg('predicted_extraversion')
                )['avg'],
                'avg_conscientiousness': profiles.aggregate(
                    avg=Avg('predicted_conscientiousness')
                )['avg'],
                'count': profiles.count(),
            }
            context['responsible_ai'] = {
                'total_models_trained': LASSO_MODEL.objects.filter(
                    volunteer__researcher=self.request.user
                ).count(),
                'total_predictions': profiles.count(),
                'avg_mae': profiles.aggregate(
                    avg=Avg('overall_mae')
                )['avg'],
            }

        return context


class ReportsView(LoginRequiredMixin, TemplateView):
    """Generate and export reports in PDF format."""
    template_name = 'dashboard/reports.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        volunteers = VOLUNTEER.objects.filter(researcher=self.request.user)

        context['total_volunteers'] = volunteers.count()
        context['volunteers'] = volunteers.order_by('-created_at')[:20]
        context['export_formats'] = ['PDF', 'CSV', 'JSON']

        return context
