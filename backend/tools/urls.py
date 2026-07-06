"""Tools and data management URLs."""
from django.urls import path
from . import views

app_name = 'tools'

urlpatterns = [
    path('', views.ToolsView.as_view(), name='index'),
    path('csv-upload/', views.CSVUploadView.as_view(), name='csv_upload'),
    path('fetch-posts/<int:volunteer_id>/', views.FetchPostsView.as_view(), name='fetch_posts'),
    path('run-pipeline/<int:volunteer_id>/', views.RunPipelineView.as_view(), name='run_pipeline'),
]
