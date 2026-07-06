#!/bin/bash
set -e

echo "Starting Django application..."

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput || echo "Migration warning (may be expected on first run)"

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || echo "Static files collection skipped"

# Create superuser if it doesn't exist
echo "Setting up admin user..."
python manage.py shell << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
    print('Admin user created: username=admin, password=admin123')
else:
    print('Admin user already exists')
END

echo "Starting Gunicorn..."
exec "$@"
