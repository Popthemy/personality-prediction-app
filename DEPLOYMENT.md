# Deployment and Setup Guide

## Local Development Setup

### 1. Environment Variables

Copy `.env.example` to `.env.local` and configure:

```bash
cp .env.example .env.local
```

Configure your `.env.local`:
```
ENVIRONMENT=development
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost/bfi_db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb bfi_db

# Apply migrations
python manage.py migrate
```

### 3. Create Superuser

```bash
python manage.py createsuperuser
```

### 4. Run Development Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Django dev server (port 8000)
python manage.py runserver

# In another terminal, start Celery worker
celery -A backend.config worker -l info

# In another terminal, start Redis server
redis-server
```

Access the application:
- Home: http://localhost:8000/
- Admin: http://localhost:8000/admin/
- Dashboard: http://localhost:8000/accounts/login/

## Production Deployment

### 1. Environment Configuration

Create `.env.prod` with production settings:
```
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=generate-long-random-string
DATABASE_URL=postgresql://prod_user:pwd@db-host/bfi_prod
REDIS_URL=redis://redis-host:6379/0
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

### 2. Run Migrations

```bash
python manage.py migrate --settings=backend.config.settings.prod
```

### 3. Collect Static Files

```bash
python manage.py collectstatic --noinput --settings=backend.config.settings.prod
```

### 4. Gunicorn Server

```bash
gunicorn backend.config.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class sync \
  --timeout 120
```

### 5. Celery Worker (Background Tasks)

```bash
celery -A backend.config worker -l info --concurrency=4
```

### 6. Nginx Configuration (Example)

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /static/ {
        alias /path/to/project/backend/static/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run migrations and collect static
RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "backend.config.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: bfi_db
      POSTGRES_USER: bfi_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  web:
    build: .
    command: gunicorn backend.config.wsgi:application --bind 0.0.0.0:8000
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://bfi_user:secure_password@db:5432/bfi_db
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery:
    build: .
    command: celery -A backend.config worker -l info
    environment:
      DATABASE_URL: postgresql://bfi_user:secure_password@db:5432/bfi_db
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

Run with: `docker-compose up -d`

## Database Models

The system uses 8 core models:

1. **VOLUNTEER**: Research participant metadata
2. **BFI_SURVEY**: Ground truth Big Five personality scores
3. **POST**: Social media text samples
4. **Q_LEARNING_LOG**: Active learning decisions
5. **BERT_EMBEDDING**: Contextual text embeddings (768-dim)
6. **SYNTHETIC_DATA**: GAN-augmented samples
7. **LASSO_MODEL**: Per-trait regression models
8. **PSYCHOMETRIC_PROFILE**: Final predictions and metrics

## Pipeline Execution

The ML pipeline follows a strict 4-stage sequence:

```
Input Data (CSV/API)
    ↓
1. Q-Learning Active Signal Selection
    ↓
2. BERT Contextual Embedding (768-dim)
    ↓
3. GAN Data Augmentation
    ↓
4. Lasso Regression (5 trait models)
    ↓
OCEAN Score Output + Performance Metrics
```

## Running the Pipeline

```python
# Trigger async pipeline execution
from backend.ml_pipeline.tasks import run_full_pipeline_task

# Queue the task
run_full_pipeline_task.delay(volunteer_id=123)

# Monitor progress in Django admin
```

## API Endpoints

- `POST /api/predict/` - Live personality prediction from text
- `GET /dashboard/` - Research dashboard
- `POST /tools/csv-upload/` - Import BFI-44 survey data
- `POST /tools/run-pipeline/<volunteer_id>/` - Execute full pipeline

## Monitoring and Logging

- Django logs: `/logs/django.log`
- Celery logs: `/logs/celery.log`
- Error logs: `/logs/error.log`

Check logs with:
```bash
tail -f logs/django.log
```

## Performance Optimization

- Static files served via WhiteNoise
- Database connection pooling configured
- Redis caching for query optimization
- Async Celery tasks for ML operations
- BERT model cached after first load

## Security Checklist

- [ ] Set `DEBUG=False` in production
- [ ] Generate strong `SECRET_KEY`
- [ ] Configure ALLOWED_HOSTS
- [ ] Use HTTPS only
- [ ] Enable CSRF protection (enabled by default)
- [ ] Configure CORS headers if API is accessed from other domains
- [ ] Use environment variables for secrets
- [ ] Enable database backups
- [ ] Set up monitoring and alerts

## Support and Documentation

See `README.md` for project overview and architecture.
