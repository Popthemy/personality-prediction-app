# Quick Start Guide

## 30-Second Setup

```bash
# 1. Run the startup script
./runserver.sh

# 2. In another terminal, start Celery
source .venv/bin/activate
celery -A backend.config worker -l info

# 3. In another terminal, start Redis
redis-server

# 4. Visit http://localhost:8000/
```

## What This Project Does

This is a production-ready Django application that predicts the Big Five personality traits from social media text using:

- **BERT**: 768-dimensional contextual embeddings
- **Q-Learning**: Intelligent post selection
- **GANs**: Synthetic data augmentation
- **Lasso Regression**: Interpretable per-trait predictions
- **BFI-44**: Ground truth validation

## Key Features

### For Researchers
- Import volunteer data via CSV (BFI-44 ground truth)
- Execute full 4-stage ML pipeline with one click
- View detailed personality profiles with OCEAN radar charts
- Track model performance metrics (MAE, correlation, R²)
- Domain-specific insights (Education, Health, Employment, Responsible AI)

### For Public Users
- Live personality prediction tool
- Enter text or Twitter handle for instant analysis
- View OCEAN scores with interactive radar chart
- Get domain-specific recommendations

## Project Structure

```
backend/
├── config/              # Django settings & WSGI
│   ├── settings/
│   │   ├── base.py     # Base configuration
│   │   ├── dev.py      # Development overrides
│   │   └── prod.py     # Production overrides
│   ├── urls.py         # Main URL routing
│   ├── wsgi.py
│   └── celery.py       # Celery async tasks
├── core/               # Core app (models, BFI scoring)
│   ├── models.py       # 8 core models
│   └── services/
│       └── bfi_scorer.py  # BFI-44 calculator
├── ml_pipeline/
    │
    ├── cleaning/
    │   └── cleaner.py
    │
    ├── processors/
    │   ├── base.py
    │   ├── profile.py
    │   ├── linguistic.py
    │   ├── temporal.py
    │   ├── engagement.py
    │   ├── network.py
    │   └── sentiment.py
    │
    ├── aggregation/
    │   └── aggregator.py
    │
    ├── services/
    │   ├── qlearning_agent.py
    │   ├── selection_collector.py
    │   ├── bert_encoder.py
    │   ├── gan_augmenter.py
    │   ├── lasso_regressor.py
    │   ├── pipeline_orchestrator.py
    │   └── insight_engine.py
    │
    └── tests/
│   └── tasks.py        # Celery tasks
├── dashboard/          # Researcher dashboard
├── tools/              # CSV import & pipeline trigger
├── accounts/           # Authentication
├── public/             # Landing page & live prediction
└── templates/          # HTML templates with Tailwind CSS

manage.py              # Django management
requirements.txt       # Python dependencies
runserver.sh          # Startup script
```

## Database Models

### 1. VOLUNTEER
```python
- id, twitter_handle, researcher (FK)
- created_at, updated_at
```

### 2. BFI_SURVEY (Ground Truth)
```python
- volunteer (FK), researcher (FK)
- responses (JSON: {q1: 4, q2: 3, ...})
- openness, conscientiousness, extraversion, agreeableness, neuroticism
```

### 3. POST
```python
- volunteer (FK), text
- created_at, retweets, likes
```

### 4. Q_LEARNING_LOG
```python
- volunteer (FK), decision, details (JSON)
- q_value, reward
```

### 5. BERT_EMBEDDING
```python
- volunteer (FK), embedding_vector (768 dimensions)
- model_name, dimension
```

### 6. SYNTHETIC_DATA
```python
- volunteer (FK), synthetic_embedding
- generation_method, confidence_score
```

### 7. LASSO_MODEL
```python
- volunteer (FK), trait
- model_coefficients, feature_importance
- r2_score, alpha
```

### 8. PSYCHOMETRIC_PROFILE (Final Predictions)
```python
- volunteer (FK), openness/conscientiousness/etc (predicted)
- openness/conscientiousness/etc (ground truth)
- mae_score, correlation, r2_score, confidence_score
```

## ML Pipeline Flow

```
STAGE 1: Q-Learning Active Signal Selection
├─ Input: Posts for a volunteer
├─ Process: Epsilon-greedy exploration
└─ Output: Selected high-value posts

STAGE 2: BERT Contextual Embedding
├─ Input: Selected post texts
├─ Process: Forward pass through bert-base-uncased
└─ Output: 768-dimensional embeddings

STAGE 3: GAN Data Augmentation
├─ Input: BERT embeddings
├─ Process: Generator network creates synthetic samples
└─ Output: Augmented embedding space

STAGE 4: Lasso Regression Prediction
├─ Input: Original + augmented embeddings, BFI ground truth
├─ Process: Train 5 separate L1-regularized models (one per trait)
└─ Output: OCEAN scores + feature importance
```

## API Endpoints

### Live Prediction
```bash
POST /api/predict/
Content-Type: application/json

{
  "text": "Your text sample here...",
  "twitter_handle": "@username"  # optional
}

Response:
{
  "status": "success",
  "ocean_scores": {
    "openness": 3.5,
    "conscientiousness": 3.2,
    "extraversion": 3.0,
    "agreeableness": 3.3,
    "neuroticism": 2.8
  },
  "confidence": 0.82,
  "domain_insights": {...}
}
```

### Trigger Pipeline
```bash
POST /tools/run-pipeline/123/
(authenticated, Form submission)

Queues: backend.ml_pipeline.tasks.run_full_pipeline_task
```

## Common Tasks

### Import Volunteer Data
1. Go to http://localhost:8000/tools/csv-upload/
2. Upload CSV with columns: `twitter_handle, q1, q2, ..., q44`
3. System calculates BFI-44 scores and stores ground truth

### Run Full Pipeline
1. Dashboard → Tools → Run ML Pipeline
2. Select volunteer with completed BFI survey
3. Monitor progress in terminal
4. View results in Dashboard → Volunteer Profile

### Check Admin Interface
1. http://localhost:8000/admin/
2. Username: admin (from `createsuperuser`)
3. Browse all models, edit volunteer data

### View Live Predictions
1. http://localhost:8000/live-prediction/
2. Enter text or Twitter handle
3. Get instant OCEAN scores and insights

## Environment Variables

Critical variables in `.env.local`:

```
ENVIRONMENT=development              # or production
DEBUG=True                           # False in production
SECRET_KEY=django-insecure-...       # Generate with: python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
DATABASE_URL=postgresql://user:pw@localhost/bfi_db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1
```

## Troubleshooting

### "ModuleNotFoundError: No module named..."
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "psycopg2 connection refused"
- Ensure PostgreSQL is running: `psql --version`
- Check DATABASE_URL in `.env.local`
- Create database: `createdb bfi_db`

### Celery tasks not executing
- Check Redis is running: `redis-cli ping` (should return PONG)
- Check Celery worker logs for errors
- Verify CELERY_BROKER_URL in `.env.local`

### BERT model not loading
- First load takes ~1 minute to download bert-base-uncased
- Check internet connection
- Model cached in `~/.cache/huggingface/hub/`

### Templates not found
```bash
python manage.py collectstatic --settings=backend.config.settings.dev
```

## Performance Tips

1. **Run migrations**: Use `--noinput` flag in production
2. **Cache BERT model**: Loads only once, then cached
3. **Batch processing**: Queue multiple pipeline tasks together
4. **Database indexing**: BFI_SURVEY and POST are indexed on volunteer_id
5. **Static files**: Served via WhiteNoise in production

## Security Notes

- CSRF protection enabled by default
- SQL injection prevented via ORM
- Password hashing via Django's PBKDF2
- Environment variables for secrets (not hardcoded)
- Row-level security: Users see only their own data

## Next Steps

1. **Customize BFI Scoring**: Edit `backend/core/services/bfi_scorer.py`
2. **Add X API Integration**: Implement in `backend/tools/views.py`
3. **Fine-tune BERT**: Create custom LoRA adapter in `bert_encoder.py`
4. **Optimize Lasso**: Adjust alpha/regularization in `lasso_regressor.py`
5. **Add Domain Models**: Create specialized models per domain

## Documentation

- `README.md` - Project overview and architecture
- `DEPLOYMENT.md` - Production setup and Docker
- `QUICKSTART.md` - This file
- Code comments throughout all services

## Support

For issues or questions:
1. Check logs: `tail -f logs/django.log`
2. Review Django admin: http://localhost:8000/admin/
3. Test API: Use provided curl examples
4. Check status of background tasks: Celery logs
