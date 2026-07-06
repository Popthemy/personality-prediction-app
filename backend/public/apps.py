from django.apps import AppConfig


class PublicConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'backend.public'
    verbose_name = 'Public Pages'
