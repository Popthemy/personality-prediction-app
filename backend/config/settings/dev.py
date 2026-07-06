"""
Development settings for Big Five Personality Prediction System.
"""
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '*']

# SQLite for development (override base PostgreSQL)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Development logging
LOGGING['loggers']['django']['level'] = 'DEBUG'
LOGGING['handlers']['console']['level'] = 'DEBUG'

# CORS for development
CORS_ALLOW_ALL_ORIGINS = True

# Simple cache for development
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Celery for development (synchronous)
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
