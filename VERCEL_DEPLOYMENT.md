# Vercel Deployment Guide - Big Five Personality Prediction System

## Overview

This Django application is containerized and ready for deployment on Vercel using their container/Docker runtime. The deployment includes:

- **Dockerfile**: Production-grade Django container with Gunicorn
- **docker-compose.yml**: Local testing with PostgreSQL, Redis, and Celery
- **vercel.json**: Vercel-specific configuration
- **entrypoint.sh**: Container initialization script

## Prerequisites

1. **Vercel Account**: Create one at https://vercel.com
2. **GitHub Repository**: This project should be in a GitHub repo
3. **Environment Variables**: Configure these in Vercel dashboard

## Deployment Steps

### Step 1: Push to GitHub

```bash
cd /vercel/share/v0-project
git add .
git commit -m "Add Docker and Vercel deployment configuration"
git push origin main
```

### Step 2: Connect to Vercel

1. Go to https://vercel.com/new
2. Select "Other" (since this is a custom Docker setup)
3. Connect your GitHub repository
4. Select this project

### Step 3: Configure Environment Variables

In the Vercel Dashboard, set these environment variables:

**Required:**
```
DJANGO_SETTINGS_MODULE=backend.config.settings.prod
DEBUG=False
SECRET_KEY=<generate-with-django>
ALLOWED_HOSTS=your-app.vercel.app,your-domain.com
```

**Optional (for full features):**
```
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://user:pass@host:6379/0
CORS_ALLOWED_ORIGINS=https://your-domain.com
```

**Generate SECRET_KEY:**
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### Step 4: Deploy

Once environment variables are set, click "Deploy". Vercel will:

1. Build the Docker image
2. Run migrations (via entrypoint.sh)
3. Collect static files
4. Create admin user
5. Start Gunicorn server

### Step 5: Verify Deployment

After deployment completes:

```bash
# Check health endpoint
curl https://your-app.vercel.app/health/

# Expected response:
# {"status": "healthy", "database": "ok"}
```

Access the app:
- Home: https://your-app.vercel.app/
- Live Prediction: https://your-app.vercel.app/live-prediction/
- Admin: https://your-app.vercel.app/admin/ (credentials: admin/admin123)

## Local Testing with Docker Compose

Before deploying to Vercel, test locally:

```bash
# Build and run all services
docker-compose up -d

# Check logs
docker-compose logs -f web

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# Access locally
# http://localhost:8000/
# http://localhost:8000/admin/

# Stop services
docker-compose down
```

## Environment Variables Explained

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `DJANGO_SETTINGS_MODULE` | Which settings to use | Yes | `backend.config.settings.prod` |
| `DEBUG` | Debug mode (ALWAYS False in production) | Yes | `False` |
| `SECRET_KEY` | Django secret key (generate new) | Yes | `<random-string>` |
| `ALLOWED_HOSTS` | Allowed domains | Yes | `app.vercel.app,mysite.com` |
| `DATABASE_URL` | PostgreSQL connection | No* | `postgresql://user:pass@host/db` |
| `REDIS_URL` | Redis for Celery | No* | `redis://host:6379/0` |

*Without these, the app will use SQLite and disable background tasks.

## Production Considerations

### Database

For production, use PostgreSQL (SQLite won't persist on Vercel):

1. Use Vercel's PostgreSQL integration, or
2. Use a managed service like:
   - Neon (free tier available)
   - Railway
   - Amazon RDS
   - DigitalOcean

3. Set `DATABASE_URL` environment variable

### Redis/Celery

For background tasks (ML pipeline execution):

1. Use a managed Redis service:
   - Upstash (Vercel-friendly, free tier)
   - Redis Labs
   - AWS ElastiCache

2. Set `REDIS_URL` environment variable

### Static Files

Static files are collected during build and served via WhiteNoise middleware. No additional setup needed.

### SSL/HTTPS

Vercel automatically provides SSL certificates. HTTPS is enforced via `SECURE_SSL_REDIRECT=True` in production settings.

## Troubleshooting

### Deployment Fails

Check logs in Vercel dashboard:
1. Go to your project
2. Click "Deployments"
3. Select latest deployment
4. View "Build Logs" and "Runtime Logs"

Common issues:
- **Database error**: `DATABASE_URL` not set
- **Secret key error**: `SECRET_KEY` not set
- **Static files error**: Usually resolved automatically

### App Returns 500 Error

1. Check runtime logs in Vercel dashboard
2. Verify all environment variables are set
3. Check database connectivity: `curl https://app.vercel.app/health/`

### Admin Login Not Working

Default credentials are set in `entrypoint.sh`:
- Username: `admin`
- Password: `admin123`

Change password immediately after login.

## File Structure

```
.
├── Dockerfile              # Production container definition
├── .dockerignore          # Files to exclude from Docker build
├── docker-compose.yml     # Local development services
├── entrypoint.sh         # Container startup script
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── manage.py            # Django management
└── backend/
    ├── config/
    │   ├── settings/
    │   │   ├── base.py     # Base settings
    │   │   ├── dev.py      # Development
    │   │   └── prod.py     # Production (for Vercel)
    │   ├── wsgi.py        # WSGI app for Gunicorn
    │   └── celery.py      # Celery configuration
    └── [other apps]
```

## Next Steps

1. **Generate a new SECRET_KEY** (don't use default)
2. **Set up PostgreSQL** if not using SQLite
3. **Set up Redis** for background tasks (optional)
4. **Deploy to Vercel** following the steps above
5. **Test all features** in production
6. **Set up monitoring** (optional: Sentry, DataDog)

## Support

For deployment issues:
- Vercel Docs: https://vercel.com/docs
- Django Deployment: https://docs.djangoproject.com/en/5.0/howto/deployment/
- Container Debugging: Check `docker-compose logs -f`

## Security Notes

⚠️ **CRITICAL for Production:**

1. Generate a new `SECRET_KEY` (don't hardcode it)
2. Set `DEBUG=False` (verified in prod.py)
3. Use strong database passwords
4. Enable SSL/HTTPS (automatic on Vercel)
5. Set up CORS properly for your domains
6. Rotate admin credentials immediately
7. Use environment variables for all secrets

---

**Ready to deploy!** Follow the steps above to get your app live on Vercel.
