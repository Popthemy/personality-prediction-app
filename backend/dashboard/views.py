"""Dashboard views with analytics and personality insights."""
import logging
from django.views.generic import TemplateView, DetailView, ListView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Avg, Count
from backend.core.models import (
    VOLUNTEER, PSYCHOMETRIC_PROFILE, BFI_SURVEY, 
    LASSO_MODEL, Q_LEARNING_LOG
)

logger = logging.getLogger(__name__)


class DashboardView(LoginRequiredMixin, TemplateView):
    """Main researcher dashboard with volunteer management and summary stats."""
    template_name = 'dashboard/index.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        volunteers = VOLUNTEER.objects.filter(researcher=self.request.user)
        
        context['total_volunteers'] = volunteers.count()
        context['volunteers_with_bfi'] = volunteers.filter(
            bfi_surveys__isnull=False
        ).distinct().count()
        context['volunteers_with_predictions'] = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer__researcher=self.request.user
        ).values('volunteer').distinct().count()
        
        # Average performance metrics
        profiles = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer__researcher=self.request.user
        )
        if profiles.exists():
            context['avg_mae'] = profiles.aggregate(
                avg_mae=Avg('mae_score')
            )['avg_mae']
            context['avg_correlation'] = profiles.aggregate(
                avg_corr=Avg('correlation')
            )['avg_corr']
        
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
        bfi_survey = volunteer.bfi_surveys.first()
        print(f"#### Volunteer: {volunteer.x_handle}, BFI Survey: {bfi_survey}")
        context['ground_truth'] = bfi_survey
        
        # Prediction profile
        profile = PSYCHOMETRIC_PROFILE.objects.filter(
            volunteer=volunteer
        ).first()
        context['prediction'] = profile
        
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
            # Domain-wise aggregation
            context['education_insights'] = {
                'avg_conscientiousness': profiles.aggregate(
                    avg=Avg('conscientiousness_predicted')
                )['avg'],
                'count': profiles.count(),
            }
            context['health_insights'] = {
                'avg_neuroticism': profiles.aggregate(
                    avg=Avg('neuroticism_predicted')
                )['avg'],
                'avg_openness': profiles.aggregate(
                    avg=Avg('openness_predicted')
                )['avg'],
                'count': profiles.count(),
            }
            context['employment_insights'] = {
                'avg_extraversion': profiles.aggregate(
                    avg=Avg('extraversion_predicted')
                )['avg'],
                'avg_conscientiousness': profiles.aggregate(
                    avg=Avg('conscientiousness_predicted')
                )['avg'],
                'count': profiles.count(),
            }
            context['responsible_ai'] = {
                'total_models_trained': LASSO_MODEL.objects.filter(
                    volunteer__researcher=self.request.user
                ).count(),
                'total_predictions': profiles.count(),
                'avg_model_accuracy': profiles.aggregate(
                    avg=Avg('r2_score')
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
