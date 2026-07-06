"""Researcher dashboard URLs."""
from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.DashboardView.as_view(), name='index'),
    path('volunteer/<int:pk>/', views.VolunteerDetailView.as_view(), name='volunteer_detail'),
    path('insights/', views.DomainInsightsView.as_view(), name='insights'),
    path('reports/', views.ReportsView.as_view(), name='reports'),
]
