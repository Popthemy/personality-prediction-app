#!/bin/bash

##############################################################################
# START DJANGO PROJECT - SIMPLE ONE-COMMAND STARTUP
# Copy & paste this entire file or run: bash START_HERE.sh
##############################################################################

set -e  # Exit on error

echo ""
echo "=========================================================================="
echo " STARTING DJANGO BIG FIVE PERSONALITY PREDICTION SYSTEM"
echo "=========================================================================="
echo ""

# Navigate to project directory
cd /vercel/share/v0-project

# Step 1: Install dependencies (if not already installed)
echo "[1/4] Installing dependencies..."
pip install -q django==5.0.6 psycopg2-binary celery redis transformers torch scikit-learn numpy pandas pillow django-environ django-cors-headers whitenoise gunicorn 2>/dev/null || true

# Step 2: Set environment variables
echo "[2/4] Configuring environment..."
export DJANGO_SETTINGS_MODULE=backend.config.settings.dev
export SECRET_KEY=django-insecure-dev-key-for-testing-only
export DEBUG=True
export ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# Step 3: Setup database
echo "[3/4] Setting up database..."
python manage.py migrate --noinput --run-syncdb 2>/dev/null || true

# Create admin user if doesn't exist
python manage.py shell << 'EOF' 2>/dev/null || true
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
EOF

# Step 4: Run server
echo "[4/4] Starting Django server..."
echo ""
echo "=========================================================================="
echo " ✓ Server starting..."
echo "=========================================================================="
echo ""
echo " ACCESS THESE URLS:"
echo ""
echo "   HOME PAGE:         http://localhost:8000/"
echo "   LIVE PREDICTION:   http://localhost:8000/live-prediction/"
echo "   DASHBOARD:         http://localhost:8000/dashboard/"
echo "   ADMIN:             http://localhost:8000/admin/"
echo ""
echo " LOGIN CREDENTIALS:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo " STOP SERVER: Press Ctrl+C"
echo ""
echo "=========================================================================="
echo ""

# Start server without reload to reduce debug spam
python manage.py runserver 0.0.0.0:8000 --noreload --nothreading
