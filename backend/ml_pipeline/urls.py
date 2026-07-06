"""ML Pipeline API URLs."""
from django.urls import path
from . import views

app_name = 'ml_pipeline'

urlpatterns = [
    path('status/<int:volunteer_id>/', views.PipelineStatusView.as_view(), name='status'),
    path('logs/<int:volunteer_id>/', views.PipelineLogsView.as_view(), name='logs'),
]
