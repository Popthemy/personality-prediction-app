#!/bin/bash
# Development server startup script

set -e

echo "=== Big Five Personality Prediction System ==="
echo "Starting development environment..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/update dependencies
if [ ! -f ".dependencies_installed" ]; then
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    touch .dependencies_installed
fi

# Check environment file
if [ ! -f ".env.local" ]; then
    echo "Creating .env.local from template..."
    cp .env.example .env.local
    echo "⚠️  Edit .env.local with your database configuration"
fi

# Create necessary directories
mkdir -p logs
mkdir -p backend/static
mkdir -p backend/media

# Run migrations
echo "Running database migrations..."
python manage.py migrate --settings=backend.config.settings.dev

# Create superuser if needed
echo ""
read -p "Create superuser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python manage.py createsuperuser --settings=backend.config.settings.dev
fi

echo ""
echo "=== Development Server Starting ==="
echo ""
echo "Access the application:"
echo "  Home:      http://localhost:8000/"
echo "  Dashboard: http://localhost:8000/accounts/login/"
echo "  Admin:     http://localhost:8000/admin/"
echo ""
echo "To run Celery worker in another terminal:"
echo "  source .venv/bin/activate"
echo "  celery -A backend.config worker -l info"
echo ""
echo "To run Redis in another terminal:"
echo "  redis-server"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Django development server
python manage.py runserver --settings=backend.config.settings.dev
