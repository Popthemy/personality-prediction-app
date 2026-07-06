"""ML Pipeline views."""
from django.views.generic import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse


class PipelineStatusView(LoginRequiredMixin, View):
    def get(self, request, volunteer_id):
        return JsonResponse({'status': 'pending'})


class PipelineLogsView(LoginRequiredMixin, View):
    def get(self, request, volunteer_id):
        return JsonResponse({'logs': []})
