# Big Five Personality Prediction System

A production-grade Django web application for integrating BERT, Lasso Regression, GANs, and Q-Learning to predict Big Five personality traits from social media-style text.

## Project Overview

**Title:** Integrating BERT, Lasso Regression, GANs, and Q-Learning for Big Five Personality Prediction from Social Media-Style Text

This system demonstrates an end-to-end ML research pipeline with equal emphasis on four domains:
- **Education**: Learning style optimization based on personality traits
- **Health & Wellbeing**: Personalized wellness recommendations  
- **Employment**: Career fit analysis and workplace culture alignment
- **Responsible AI**: Ethical considerations and fairness in AI-driven assessment

## Core Architecture

### Technology Stack
- **Framework**: Django 5.x (MVT architecture)
- **Database**: PostgreSQL (production) / SQLite (development)
- **Task Queue**: Celery + Redis
- **ML Libraries**: PyTorch, HuggingFace Transformers, scikit-learn
- **Frontend**: HTMX + Tailwind CSS
- **Visualization**: Chart.js for OCEAN radar charts

### ML Pipeline (Strict Sequential Order)
1. **Input Data**: Social media posts (X API or CSV import)
2. **Q-Learning Active Selection**: Intelligent post selection based on engagement and value
3. **BERT Embedding Extraction**: 768-dimensional contextual embeddings
4. **GAN Data Augmentation**: Synthetic training data generation
5. **Lasso Regression**: Sparse, interpretable trait prediction models

### Database Schema (8 Core Models)

```
VOLUNTEER
├── BFI_SURVEY (ground truth Big Five Inventory)
├── POST (social media posts)
│   └── BERT_EMBEDDING
│       └── SYNTHETIC_DATA (GAN-generated)
├── Q_LEARNING_LOG
├── LASSO_MODEL (one per OCEAN trait)
└── PSYCHOMETRIC_PROFILE (final predictions)
```

## Project Structure

```
/backend/
├── config/              # Django settings (dev/prod separation)
├── core/               # Core models, BFI-44 scorer
│   └── services/
│       └── bfi_scorer.py
├── ml_pipeline/        # ML pipeline execution
│   └── services/
│       ├── qlearning_agent.py
│       ├── bert_encoder.py
│       ├── gan_augmenter.py
│       ├── lasso_regressor.py
│       └── pipeline_orchestrator.py
├── accounts/           # User authentication
├── dashboard/          # Researcher dashboard
├── tools/              # Data import & pipeline control
├── public/             # Public-facing pages
├── templates/          # HTML templates (HTMX-enabled)
└── static/             # CSS, JS, images

manage.py              # Django entry point
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
cd /vercel/share/v0-project
uv venv --python 3.11
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Or manually install key packages:
pip install django==5.0.6 django-environ psycopg2-binary celery redis \
    transformers torch scikit-learn pandas numpy pillow
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env.local

# Edit .env.local with your settings
# For development: use SQLite, localhost Redis
```

### 4. Database Setup
```bash
python manage.py migrate --run-syncdb
python manage.py createsuperuser  # Create admin account
```

### 5. Run Development Server
```bash
python manage.py runserver 8000
```

Visit `http://localhost:8000`
- Public pages: `/`
- Django Admin: `/admin` (login with superuser credentials)

## Key Features

### 1. BFI-44 Scoring (`core/services/bfi_scorer.py`)
- Handles all 44 items with proper reverse-scoring
- Calculates OCEAN trait scores (1-5 scale)
- Validates responses and handles missing data

**Usage:**
```python
from backend.core.services import score_bfi_survey

responses = {"1": 4, "2": 3, ..., "44": 5}
ocean_scores = score_bfi_survey(responses)
# Returns: {'Openness': 3.5, 'Conscientiousness': 4.1, ...}
```

### 2. Q-Learning Active Selection (`ml_pipeline/services/qlearning_agent.py`)
- Learns which posts are most valuable for prediction
- Features: engagement score, recency, text length, metadata
- State space discretization and epsilon-greedy exploration
- Saves Q-values to database for interpretability

**Usage:**
```python
from backend.ml_pipeline.services import QLearningAgent, create_post_features

agent = QLearningAgent(alpha=0.1, gamma=0.99)
selected_posts = agent.select_posts(posts_with_features, top_k=10)
```

### 3. BERT Embedding Extraction (`ml_pipeline/services/bert_encoder.py`)
- Uses `bert-base-uncased` from HuggingFace
- Extracts 768-dimensional [CLS] token embeddings
- GPU-accelerated if available, falls back to CPU
- Stores embeddings as JSON in database

**Usage:**
```python
from backend.ml_pipeline.services import BERTEncoder

encoder = BERTEncoder()
result = encoder.encode_text("Your text here")
# Returns: {'embedding': [...768 values...], 'model_name': 'bert-base-uncased', ...}
```

### 4. GAN Data Augmentation (`ml_pipeline/services/gan_augmenter.py`)
- Perturbs embeddings with Gaussian noise
- Generates synthetic post text using templates
- Creates 1:1 augmentation ratio by default
- Tracks confidence scores for synthetic samples

**Usage:**
```python
from backend.ml_pipeline.services import GANAugmenter

augmenter = GANAugmenter(augmentation_factor=0.8)
synthetic_posts = augmenter.augment_training_set(posts, target_size=100)
```

### 5. Lasso Regression (`ml_pipeline/services/lasso_regressor.py`)
- Trains one sparse model per OCEAN trait
- L1 regularization for feature selection
- Returns feature importance (non-zero coefficients)
- Calculates MAE, RMSE, and R² metrics

**Usage:**
```python
from backend.ml_pipeline.services import LassoTrainer

trainer = LassoTrainer(alpha=0.001)
metrics = trainer.train_trait_model(X_train, y_train, 'Openness')
predictions = trainer.predict_all_traits(X_test)
```

### 6. Pipeline Orchestrator (`ml_pipeline/services/pipeline_orchestrator.py`)
- Executes full 5-step pipeline for a volunteer
- Manages database transactions and logging
- Saves all artifacts (embeddings, models, predictions)
- Handles errors gracefully

**Usage:**
```python
from backend.ml_pipeline.services import PipelineOrchestrator

orchestrator = PipelineOrchestrator(volunteer_id=1)
result = orchestrator.run_full_pipeline()
# Executes: Input → Q-Learn → BERT → GAN → Lasso
```

## Database Models

### VOLUNTEER
- X handle, user ID, researcher (FK)
- Demographics (age, country, language)
- Pipeline status tracking
- Consent management

### BFI_SURVEY
- 44 item responses (JSON)
- Calculated OCEAN scores (1-5)
- Completion timestamp

### POST
- Post content and metadata
- Engagement metrics (likes, retweets, replies)
- Q-Learning selection status and Q-value
- BERT processing flag

### BERT_EMBEDDING
- 768-dimensional embedding vector (JSON)
- Model name and processing time
- Post reference

### Q_LEARNING_LOG
- Episode and action tracking
- State, reward, and Q-value updates
- Learning parameters (α, γ)

### SYNTHETIC_DATA
- Synthetic text and embedding
- Generator version and confidence
- Training flag

### LASSO_MODEL
- Model coefficients (JSON)
- Trait-specific (one per trait)
- Training/validation metrics
- Feature importance

### PSYCHOMETRIC_PROFILE
- Predicted OCEAN scores
- MAE for each trait and overall
- Domain recommendations (JSON)
- Prediction confidence

## Admin Interface

Access `/admin` with superuser credentials to:
- View and manage volunteers
- Edit BFI surveys and calculate scores
- Review posts and Q-Learning decisions
- Inspect BERT embeddings
- Monitor Lasso training
- View final personality profiles
- Export data for analysis

## Logging

Logs are stored in `/logs/`:
- `django.log`: General application logs
- `ml_pipeline.log`: ML pipeline execution traces

Configure logging in `backend/config/settings/base.py`.

## Environment Configuration

### Development (.env.local)
```
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
DATABASE_URL=sqlite:////path/to/db.sqlite3
CELERY_TASK_ALWAYS_EAGER=True  # Synchronous for testing
```

### Production (.env)
```
DEBUG=False
DATABASE_URL=postgresql://user:pass@host:5432/db
CELERY_TASK_ALWAYS_EAGER=False  # Async with Redis
SECURE_SSL_REDIRECT=True
```

## API Endpoints

### Public
- `GET /` - Landing page
- `GET /live-prediction/` - Public demo
- `POST /api/predict/` - Live prediction endpoint

### Authenticated (Researcher)
- `GET /dashboard/` - Researcher dashboard
- `GET /dashboard/volunteer/<id>/` - Volunteer detail
- `GET /tools/` - Data management tools
- `POST /tools/csv-upload/` - CSV import
- `POST /tools/run-pipeline/<id>/` - Execute full pipeline

## Training a Model

### Step-by-Step

1. **Add Volunteer with Posts**
   ```bash
   # Create volunteer in admin or via ORM
   volunteer = VOLUNTEER.objects.create(
       x_handle='@username',
       researcher=user,
       consent_given=True
   )
   
   # Import posts from X API or CSV
   # Each POST object is created with engagement metrics
   ```

2. **Enter BFI-44 Ground Truth**
   ```bash
   # Admin interface or ORM
   bfi_survey = BFI_SURVEY.objects.create(
       volunteer=volunteer,
       responses={"1": 4, "2": 3, ..., "44": 5},
       completed_at=now()
   )
   # Scores are auto-calculated via signal or service
   ```

3. **Run Full Pipeline**
   ```bash
   orchestrator = PipelineOrchestrator(volunteer_id=volunteer.id)
   result = orchestrator.run_full_pipeline()
   
   # Pipeline automatically:
   # 1. Fetches posts from DB
   # 2. Q-Learning selects top 10
   # 3. BERT encodes each post
   # 4. GAN creates synthetic samples
   # 5. Lasso trains and predicts
   # 6. Saves PSYCHOMETRIC_PROFILE
   ```

4. **Evaluate Results**
   ```bash
   profile = PSYCHOMETRIC_PROFILE.objects.get(volunteer=volunteer)
   print(f"Predicted Openness: {profile.predicted_openness:.2f}")
   print(f"Ground Truth: {profile.volunteer.bfi_survey.openness:.2f}")
   print(f"MAE: {profile.mae_openness:.4f}")
   ```

## Development Tips

### Adding a New Feature
1. Create model in appropriate `models.py`
2. Register with Django admin
3. Create form if needed
4. Add view and URL pattern
5. Create template
6. Update tests/documentation

### Testing ML Services
```python
from backend.ml_pipeline.services import *

# Test Q-Learning
agent = QLearningAgent()
posts = [{'id': 1, 'engagement_score': 100, ...}]
selected = agent.select_posts(posts)

# Test BERT
encoder = BERTEncoder()
emb = encoder.encode_text("Sample text")

# Test Lasso
trainer = LassoTrainer()
trainer.train_trait_model(X, y, 'Openness')
```

## Performance Considerations

- **BERT Encoding**: ~1-2 seconds per post (GPU: ~0.5s)
- **Lasso Training**: <1 second for 100+ samples
- **Full Pipeline**: ~1 minute for 50 posts (includes BERT)

Use Celery for background processing in production.

## Security Notes

- Change `SECRET_KEY` in production
- Use PostgreSQL instead of SQLite
- Enable `SECURE_SSL_REDIRECT=True`
- Set strong database passwords
- Use environment variables for sensitive data
- Never commit `.env` to version control

## Contributing

Follow these principles:
- Modular services for reusability
- Comprehensive logging with contextual info
- Type hints for clarity
- Docstrings for all public functions
- Separation of concerns (models, views, services)

## License

Research purposes only. See LICENSE file for details.

## Support

For issues or questions:
1. Check logs in `/logs/`
2. Review admin interface
3. Test ML services in isolation
4. Consult inline code documentation

---

**Built with Django 5.x, PyTorch, and HuggingFace Transformers**
