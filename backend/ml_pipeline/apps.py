from django.apps import AppConfig


class MlPipelineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend.ml_pipeline'
    verbose_name = 'ML Pipeline'
