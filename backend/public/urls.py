"""
Public pages URLs.
"""
from django.urls import path
from . import views

app_name = 'public'

urlpatterns = [
    path('health/', views.HealthCheckView.as_view(), name='health'),
    path('', views.IndexView.as_view(), name='index'),
    path('live-prediction/', views.LivePredictionView.as_view(), name='live_prediction'),
    path('api/predict/', views.PredictAPIView.as_view(), name='api_predict'),
]
