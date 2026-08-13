"""Tools and data management URLs."""
from django.urls import path
from . import views

app_name = 'tools'

urlpatterns = [
    path('', views.ToolsView.as_view(), name='index'),
    path('csv-upload/', views.CSVUploadView.as_view(), name='csv_upload'),
    path('analyze/', views.AnalyzeProfileView.as_view(), name='analyze'),
    path('train/', views.CohortTrainingView.as_view(), name='train'),
    path('predict/', views.CohortPredictionView.as_view(), name='predict'),
    path('fetch-posts/<int:volunteer_id>/', views.FetchPostsView.as_view(), name='fetch_posts'),
    path('run-pipeline/<int:volunteer_id>/', views.RunPipelineView.as_view(), name='run_pipeline'),
    path('pipeline/control/', views.PipelineControlView.as_view(), name='pipeline_control'),
]
