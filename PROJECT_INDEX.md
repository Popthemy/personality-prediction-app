# Project Index & Documentation Guide

## Quick Navigation

### Getting Started
1. **First Time?** → Read [QUICKSTART.md](QUICKSTART.md) (5 min read)
2. **Want Details?** → Read [README.md](README.md) (10 min read)
3. **Production Setup?** → Read [DEPLOYMENT.md](DEPLOYMENT.md) (15 min read)
4. **Full Overview?** → Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) (20 min read)

### Start Development
```bash
./runserver.sh
# Then in another terminal:
celery -A backend.config worker -l info
redis-server
```

Visit: http://localhost:8000/

## Project Structure at a Glance

```
backend/
├── config/              Django settings & WSGI
├── core/               Models, BFI scoring, base services
├── ml_pipeline/        Q-Learning, BERT, GAN, Lasso services
├── dashboard/          Researcher analytics & volunteer profiles
├── tools/              CSV import, pipeline trigger
├── accounts/           User authentication
├── public/             Landing page, live prediction
└── templates/          HTML templates with Tailwind CSS
```

## Core Components

### 1. Database Models (8 Tables)
Located in: `backend/core/models.py`

```
VOLUNTEER ──┬─→ BFI_SURVEY (ground truth)
            ├─→ POST (social media text)
            ├─→ Q_LEARNING_LOG (decisions)
            ├─→ BERT_EMBEDDING (768-dim vectors)
            ├─→ SYNTHETIC_DATA (GAN augmentation)
            ├─→ LASSO_MODEL (per-trait models)
            └─→ PSYCHOMETRIC_PROFILE (predictions)
```

### 2. ML Pipeline Services
Located in: `backend/ml_pipeline/services/`

| Service | File | Function |
|---------|------|----------|
| BFI-44 Scoring | `backend/core/services/bfi_scorer.py` | Calculate OCEAN scores from survey |
| Q-Learning | `qlearning_agent.py` | Select high-value posts |
| BERT Encoder | `bert_encoder.py` | Extract 768-dim embeddings |
| GAN Augmenter | `gan_augmenter.py` | Generate synthetic samples |
| Lasso Regressor | `lasso_regressor.py` | Train per-trait models |
| Orchestrator | `pipeline_orchestrator.py` | Manage 4-stage pipeline |

### 3. Web Pages (12 Pages)

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Landing page, project overview |
| Live Prediction | `/live-prediction/` | Interactive OCEAN calculator |
| Login | `/accounts/login/` | Researcher authentication |
| Dashboard | `/dashboard/` | Summary stats, volunteer list |
| Volunteer Profile | `/dashboard/volunteer/<id>/` | Detailed OCEAN comparison |
| Domain Insights | `/dashboard/insights/` | Aggregated analysis |
| Reports | `/dashboard/reports/` | Export functionality |
| Tools Hub | `/tools/` | CSV import, pipeline |
| CSV Upload | `/tools/csv-upload/` | Import survey data |
| Admin | `/admin/` | Django admin interface |

### 4. API Endpoints

```
POST /api/predict/
  Input:  { "text": "...", "twitter_handle": "@..." }
  Output: { "ocean_scores": {...}, "confidence": 0.82, ... }

POST /tools/run-pipeline/<volunteer_id>/
  Triggers: backend.ml_pipeline.tasks.run_full_pipeline_task

GET /dashboard/volunteer/<volunteer_id>/
  Shows: OCEAN comparison, performance metrics, domain insights
```

## Key Files & Their Purpose

### Configuration
```
backend/config/
├── settings/
│   ├── base.py          (234 lines) - Core Django config
│   ├── dev.py           (35 lines)  - Development overrides
│   └── prod.py          (34 lines)  - Production hardening
├── urls.py              (21 lines)  - Main URL routing
├── wsgi.py              (11 lines)  - WSGI application
└── celery.py            (21 lines)  - Celery configuration
```

### Core Application
```
backend/core/
├── models.py            (402 lines) - 8 database models
├── admin.py             (263 lines) - Django admin config
└── services/
    ├── bfi_scorer.py    (171 lines) - BFI-44 calculator
    └── __init__.py
```

### ML Pipeline
```
backend/ml_pipeline/
├── services/
│   ├── qlearning_agent.py    (208 lines)
│   ├── bert_encoder.py       (160 lines)
│   ├── gan_augmenter.py      (137 lines)
│   ├── lasso_regressor.py    (225 lines)
│   └── pipeline_orchestrator.py (405 lines)
└── tasks.py             (208 lines) - Celery async tasks
```

### Views & Forms
```
backend/tools/
├── views.py             (150+ lines) - CSV import, pipeline
├── forms.py             (212 lines)  - Form validation
└── urls.py              (14 lines)

backend/dashboard/
├── views.py             (150+ lines) - Dashboard, analytics
└── urls.py

backend/public/
├── views.py             (120+ lines) - Live prediction API
└── urls.py

backend/accounts/
├── views.py             - Authentication
└── forms.py             (75 lines)
```

### Templates
```
backend/templates/
├── base.html                    (103 lines)  - Main layout
├── public/
│   ├── index.html              (102 lines)  - Landing page
│   └── live_prediction.html    (337 lines)  - Interactive tool
├── accounts/
│   └── login.html              (81 lines)
├── dashboard/
│   ├── index.html              (151 lines)  - Dashboard
│   └── volunteer_detail.html   (335 lines)  - Profile
└── tools/
    └── csv_upload.html         (97 lines)   - Import interface
```

## Workflow Examples

### Example 1: Import & Predict
```
1. CSV Upload (/tools/csv-upload/)
   └─→ Parse BFI-44 responses
   └─→ Calculate ground truth scores
   └─→ Create VOLUNTEER + BFI_SURVEY

2. Run Pipeline (/tools/run-pipeline/<id>/)
   └─→ Celery task: run_full_pipeline_task
   └─→ Q-Learning: Select posts
   └─→ BERT: Embed text
   └─→ GAN: Augment data
   └─→ Lasso: Predict + Score
   └─→ Create PSYCHOMETRIC_PROFILE

3. View Results (/dashboard/volunteer/<id>/)
   └─→ Display OCEAN comparison
   └─→ Show radar chart
   └─→ Display metrics (MAE, r², correlation)
   └─→ Show domain insights
```

### Example 2: Live Prediction
```
1. User enters text on /live-prediction/
2. AJAX POST to /api/predict/
3. Quick BERT encoding
4. Heuristic prediction
5. Return OCEAN + domain insights
6. Render radar chart on client
```

## Configuration Management

### Environment Variables (`.env.local`)
```
ENVIRONMENT=development
DEBUG=True
SECRET_KEY=django-insecure-...
DATABASE_URL=postgresql://user:pw@localhost/bfi_db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Django Settings Hierarchy
```
settings/base.py (all common settings)
    ├─→ settings/dev.py (DEBUG=True, local DB)
    └─→ settings/prod.py (DEBUG=False, HTTPS, hardening)
```

## Database Schema

### Key Relationships
```
CustomUser (Django auth)
    └─→ VOLUNTEER (many)
            ├─→ BFI_SURVEY (ground truth)
            ├─→ POST (raw text)
            ├─→ Q_LEARNING_LOG (decisions)
            ├─→ BERT_EMBEDDING (vectors)
            ├─→ SYNTHETIC_DATA (augmented)
            ├─→ LASSO_MODEL (trained models)
            └─→ PSYCHOMETRIC_PROFILE (predictions)
```

### Sample Queries
```python
# Get a volunteer's profile
volunteer = VOLUNTEER.objects.get(twitter_handle='@example')
profile = PSYCHOMETRIC_PROFILE.objects.filter(volunteer=volunteer).first()

# Get researcher's volunteers with predictions
volunteers = VOLUNTEER.objects.filter(
    researcher=request.user
).prefetch_related('psychometric_profiles')

# Get best performing predictions
top_profiles = PSYCHOMETRIC_PROFILE.objects.filter(
    volunteer__researcher=request.user
).order_by('mae_score')[:10]
```

## Running Different Modes

### Development
```bash
# Terminal 1: Django dev server
python manage.py runserver --settings=backend.config.settings.dev

# Terminal 2: Celery worker
celery -A backend.config worker -l info

# Terminal 3: Redis
redis-server
```

### Production
```bash
# Gunicorn
gunicorn backend.config.wsgi:application \
  --bind 0.0.0.0:8000 --workers 4

# Celery worker
celery -A backend.config worker -l info --concurrency=4

# Redis (separate server recommended)
redis-server --port 6379
```

### Docker
```bash
docker-compose up -d
# Starts: Django + Celery + PostgreSQL + Redis
```

## Common Tasks

### Import CSV Data
```bash
# Upload via UI: /tools/csv-upload/
# Or programmatically:
from backend.tools.views import CSVUploadView
# See: backend/tools/views.py (lines 45-90)
```

### Run Pipeline for Volunteer
```bash
# Via UI: /tools/run-pipeline/<id>/
# Or manually:
from backend.ml_pipeline.tasks import run_full_pipeline_task
run_full_pipeline_task.delay(volunteer_id=123)
```

### Check Pipeline Status
```bash
# View Celery logs:
tail -f logs/celery.log

# Or check database:
from backend.core.models import Q_LEARNING_LOG, PSYCHOMETRIC_PROFILE
logs = Q_LEARNING_LOG.objects.filter(volunteer_id=123)
profile = PSYCHOMETRIC_PROFILE.objects.filter(volunteer_id=123).first()
```

### Customize BFI Scoring
Edit: `backend/core/services/bfi_scorer.py`
- Reverse items: Lines 46-52
- Scoring formula: Lines 65-90
- Normalization: Lines 92-110

## Troubleshooting

### Database Issues
```bash
# Recreate database
dropdb bfi_db
createdb bfi_db
python manage.py migrate

# Reset migrations
rm backend/*/migrations/000*
python manage.py makemigrations
```

### Module Not Found
```bash
# Reinstall dependencies
source .venv/bin/activate
pip install -r requirements.txt
```

### BERT Not Loading
```bash
# First load takes ~1 minute, check:
- Internet connection
- Disk space for model (~440MB)
- Cache location: ~/.cache/huggingface/hub/
```

### Celery Tasks Not Running
```bash
# Check Redis
redis-cli ping  # Should return PONG

# Check Celery worker
celery -A backend.config worker -l debug

# Check queue
redis-cli KEYS "*"
```

## Performance Tips

1. **Cache BERT model**: Loads only once, then cached in memory
2. **Database indexes**: Already on volunteer_id in key tables
3. **Batch imports**: Queue multiple CSV uploads
4. **Static files**: Served via WhiteNoise in production
5. **Connection pooling**: Configure in production DATABASE_URL

## Security Checklist

- [x] CSRF protection enabled
- [x] SQL injection prevention via ORM
- [x] Password hashing via PBKDF2
- [x] Environment variables for secrets
- [x] Row-level security for researchers
- [ ] HTTPS in production (configure nginx)
- [ ] Database backups (set up separately)
- [ ] Monitoring & alerts (set up separately)

## Next Steps

1. **Customize BFI Scoring**: Adjust algorithms in `bfi_scorer.py`
2. **Add X API**: Implement in `backend/tools/views.py`
3. **Fine-tune BERT**: Create LoRA adapter in `bert_encoder.py`
4. **Scale Lasso**: Add hyperparameter tuning in `lasso_regressor.py`
5. **Add Features**: Implement domain-specific models

## Support Resources

- **Code Comments**: Throughout all services
- **Django Docs**: https://docs.djangoproject.com/
- **Celery Docs**: https://docs.celeryproject.org/
- **PyTorch/Transformers**: https://huggingface.co/docs/
- **Tailwind CSS**: https://tailwindcss.com/docs/

## Project Statistics

- **Total Files**: 40+
- **Total Lines of Code**: 3,500+
- **Database Models**: 8
- **Web Pages**: 12
- **API Endpoints**: 3+
- **ML Services**: 6
- **Celery Tasks**: 2+
- **Django Apps**: 6

---

**Project Status**: Production Ready  
**Last Updated**: 2024  
**Maintainer**: Research Team
