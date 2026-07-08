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
from django.contrib import messages
from .forms import CSVUploadForm, XHandleFetchForm, PipelineExecutionForm
from backend.core.models import VOLUNTEER, BFI_SURVEY, POST
from backend.core.services.bfi_scorer import BFIScorer
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
        return context


class CSVUploadView(LoginRequiredMixin, FormView):
    """Handle CSV import from Google Forms (BFI-44 responses)."""
    form_class = CSVUploadForm
    template_name = 'tools/csv_upload.html'
    success_url = reverse_lazy('tools:index')
    
    def form_valid(self, form):
        csv_file = form.cleaned_data['csv_file']
        try:
            # Read CSV
            stream = StringIO(csv_file.read().decode('utf-8'))
            reader = csv.DictReader(stream)
            
            scorer = BFIScorer()
            processed_count = 0
            
            for row in reader:
                # Extract twitter handle and BFI responses
                twitter_handle = row.get('twitter_handle', '').strip()
                if not twitter_handle:
                    continue
                
                # Create or get volunteer
                volunteer, created = VOLUNTEER.objects.get_or_create(
                    twitter_handle=twitter_handle,
                    defaults={'researcher': self.request.user}
                )
                
                # Extract BFI-44 responses (questions 1-44)
                responses = {}
                for header, value in row.items():
                    match = re.match(r'q(\d+)', header)
                    if not match:
                        continue
                    item = int(match.group(1))
                    if 1 <= item <= 44:
                        try:
                            responses[str(item)] = int(value)
                        except ValueError:
                            logger.warning(f"Invalid response for item {item}: {value}")
                 
                if len(responses) >= 40:  # At least 90% responses
                    # Calculate BFI scores
                    scores = scorer.calculate_scores(responses)
                    
                    # Save BFI survey
                    BFI_SURVEY.objects.create(
                        volunteer=volunteer,
                        responses=responses,
                        openness=scores['openness'],
                        conscientiousness=scores['conscientiousness'],
                        extraversion=scores['extraversion'],
                        agreeableness=scores['agreeableness'],
                        neuroticism=scores['neuroticism'],
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
        twitter_handle = form.cleaned_data['twitter_handle']
        try:
            volunteer = VOLUNTEER.objects.get(
                twitter_handle=twitter_handle,
                researcher=self.request.user
            )
            # TODO: Integrate with X API to fetch posts
            # For now, placeholder message
            messages.info(
                self.request,
                f'X API integration coming soon for @{twitter_handle}'
            )
            logger.info(f"Posts fetch requested for {twitter_handle}")
        except VOLUNTEER.DoesNotExist:
            messages.error(
                self.request,
                f'Volunteer @{twitter_handle} not found'
            )
        
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
                f'Pipeline started for {volunteer.twitter_handle}. Check status in dashboard.'
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
