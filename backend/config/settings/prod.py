"""
Production settings for Big Five Personality Prediction System.
"""
from .base import *

DEBUG = False

# Allowed hosts - support Vercel subdomains
ALLOWED_HOSTS = env('ALLOWED_HOSTS', default='localhost,127.0.0.1').split(',')
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS]

# PostgreSQL in production
if env('DATABASE_URL', default=None):
    DATABASES = {
        'default': env.db('DATABASE_URL')
    }
else:
    # Fallback to SQLite if DATABASE_URL not set
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Logging for production
LOGGING['loggers']['django']['level'] = 'INFO'
LOGGING['handlers']['console']['level'] = 'INFO'

# CORS for production
CORS_ALLOWED_ORIGINS = env('CORS_ALLOWED_ORIGINS', default='').split(',') if env('CORS_ALLOWED_ORIGINS', default='') else []

# Security headers
SECURE_SSL_REDIRECT = env('SECURE_SSL_REDIRECT', default=True)
SESSION_COOKIE_SECURE = env('SESSION_COOKIE_SECURE', default=True)
CSRF_COOKIE_SECURE = env('CSRF_COOKIE_SECURE', default=True)
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_SECURITY_POLICY = {
    'default-src': ("'self'",),
    'script-src': ("'self'", "'unsafe-inline'", 'cdn.jsdelivr.net'),
    'style-src': ("'self'", "'unsafe-inline'", 'cdn.jsdelivr.net'),
}

# Async Celery in production
CELERY_TASK_ALWAYS_EAGER = False
CELERY_BROKER_URL = env('REDIS_URL', default='redis://localhost:6379/0')
CELERY_RESULT_BACKEND = env('REDIS_URL', default='redis://localhost:6379/0')
