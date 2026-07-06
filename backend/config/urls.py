"""
Main URL configuration for Big Five Personality Prediction System.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('backend.public.urls', namespace='public')),
    path('accounts/', include('backend.accounts.urls', namespace='accounts')),
    path('dashboard/', include('backend.dashboard.urls', namespace='dashboard')),
    path('tools/', include('backend.tools.urls', namespace='tools')),
    path('api/pipeline/', include('backend.ml_pipeline.urls', namespace='ml_pipeline')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
