"""
Core models for Big Five Personality Prediction System.

Models:
- VOLUNTEER: Research participant with X handle
- BFI_SURVEY: Ground truth Big Five Inventory responses
- POST: Social media posts from X API
- BERT_EMBEDDING: Contextual embeddings from BERT
- Q_LEARNING_LOG: Active signal selection decisions
- SYNTHETIC_DATA: GAN-generated augmented training data
- LASSO_MODEL: Trained Lasso regression models (per trait)
- PSYCHOMETRIC_PROFILE: Final predicted personality profile
"""
from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator
import json

User = get_user_model()


class VOLUNTEER(models.Model):
    """Research participant."""
    
    id = models.BigAutoField(primary_key=True)
    x_handle = models.CharField(max_length=100, unique=True, db_index=True)
    x_user_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    researcher = models.ForeignKey(User, on_delete=models.CASCADE, related_name='volunteers')
    
    # Demographics (optional)
    age = models.IntegerField(null=True, blank=True, validators=[MinValueValidator(13), MaxValueValidator(120)])
    country = models.CharField(max_length=50, blank=True)
    language = models.CharField(max_length=20, default='en')
    
    # Status
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    posts_fetched_at = models.DateTimeField(null=True, blank=True)
    pipeline_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('error', 'Error'),
        ],
        default='pending',
        db_index=True
    )
    
    # Consent & metadata
    consent_given = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    
    class Meta:
        db_table = 'volunteer'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['x_handle', 'pipeline_status']),
            models.Index(fields=['researcher', 'created_at']),
        ]
    
    def __str__(self):
        return f"@{self.x_handle}"


class BFI_SURVEY(models.Model):
    """Big Five Inventory (BFI-44) ground truth responses."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.OneToOneField(VOLUNTEER, on_delete=models.CASCADE, related_name='bfi_survey')
    
    # 44 items, each 1-5 Likert scale
    # Storage as JSON: {"1": 4, "2": 3, ...}
    responses = models.JSONField(default=dict)
    
    # Calculated trait scores (1-5 scale after reverse scoring)
    openness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    conscientiousness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    extraversion = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    agreeableness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    neuroticism = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'bfi_survey'
        verbose_name_plural = 'BFI Surveys'
    
    def __str__(self):
        return f"BFI for {self.volunteer.x_handle}"
    
    def get_ocean_dict(self):
        """Return OCEAN traits as dictionary."""
        return {
            'Openness': self.openness,
            'Conscientiousness': self.conscientiousness,
            'Extraversion': self.extraversion,
            'Agreeableness': self.agreeableness,
            'Neuroticism': self.neuroticism,
        }


class POST(models.Model):
    """Social media posts from X API."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.ForeignKey(VOLUNTEER, on_delete=models.CASCADE, related_name='posts')
    
    # Post content
    x_post_id = models.CharField(max_length=50, unique=True)
    content = models.TextField()
    created_at_original = models.DateTimeField()  # When posted on X
    
    # Quality metrics
    retweet_count = models.IntegerField(default=0)
    like_count = models.IntegerField(default=0)
    reply_count = models.IntegerField(default=0)
    
    # Engagement score for Q-Learning
    engagement_score = models.FloatField(default=0.0)
    
    # Q-Learning selection status
    selected_by_qlearning = models.BooleanField(default=False)
    q_value = models.FloatField(null=True, blank=True)  # Q(state, action) value
    
    # BERT embedding status
    embedding_processed = models.BooleanField(default=False)
    
    # Metadata
    language_detected = models.CharField(max_length=10, default='en')
    is_reply = models.BooleanField(default=False)
    is_retweet = models.BooleanField(default=False)
    
    fetched_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'post'
        ordering = ['-created_at_original']
        indexes = [
            models.Index(fields=['volunteer', 'selected_by_qlearning']),
            models.Index(fields=['volunteer', 'created_at_original']),
        ]
    
    def __str__(self):
        return f"Post by {self.volunteer.x_handle} ({self.x_post_id[:20]})"


class BERT_EMBEDDING(models.Model):
    """BERT contextual embeddings from posts."""
    
    id = models.BigAutoField(primary_key=True)
    post = models.OneToOneField(POST, on_delete=models.CASCADE, related_name='bert_embedding')
    volunteer = models.ForeignKey(VOLUNTEER, on_delete=models.CASCADE, related_name='bert_embeddings')
    
    # BERT embedding (768-dimensional)
    # Storage as JSON: {"0": 0.234, "1": -0.123, ...}
    embedding_vector = models.JSONField()
    
    # Metadata
    model_name = models.CharField(max_length=100, default='bert-base-uncased')
    processing_time_seconds = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'bert_embedding'
    
    def __str__(self):
        return f"BERT Embedding for Post {self.post.x_post_id[:20]}"


class Q_LEARNING_LOG(models.Model):
    """Q-Learning active signal selection decisions."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.ForeignKey(VOLUNTEER, on_delete=models.CASCADE, related_name='qlearning_logs')
    
    # Episode metadata
    episode = models.IntegerField()
    state = models.JSONField()  # Post features: engagement, date, content_length, etc.
    action = models.CharField(
        max_length=20,
        choices=[
            ('select', 'Select for training'),
            ('skip', 'Skip'),
        ]
    )
    reward = models.FloatField()  # Reward signal (based on engagement or downstream accuracy)
    
    # Q-Learning update
    old_q_value = models.FloatField(null=True, blank=True)
    new_q_value = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(default=0.1)
    discount_factor = models.FloatField(default=0.99)
    
    # Context
    post = models.ForeignKey(POST, on_delete=models.SET_NULL, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'q_learning_log'
        ordering = ['-episode', '-created_at']
    
    def __str__(self):
        return f"Q-Learning Episode {self.episode} for {self.volunteer.x_handle}"


class SYNTHETIC_DATA(models.Model):
    """GAN-generated augmented training data."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.ForeignKey(VOLUNTEER, on_delete=models.CASCADE, related_name='synthetic_data')
    
    # Original data reference
    original_post = models.ForeignKey(POST, on_delete=models.SET_NULL, null=True, blank=True)
    original_embedding = models.ForeignKey(BERT_EMBEDDING, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Generated content
    synthetic_text = models.TextField()
    synthetic_embedding = models.JSONField()  # 768-dimensional vector
    
    # GAN metadata
    generator_version = models.CharField(max_length=50, default='gan-v1')
    generation_confidence = models.FloatField(validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Training status
    used_in_training = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'synthetic_data'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Synthetic Data for {self.volunteer.x_handle}"


class LASSO_MODEL(models.Model):
    """Trained Lasso regression models (one per OCEAN trait)."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.ForeignKey(VOLUNTEER, on_delete=models.CASCADE, related_name='lasso_models')
    
    # Trait being predicted
    trait = models.CharField(
        max_length=20,
        choices=[
            ('openness', 'Openness'),
            ('conscientiousness', 'Conscientiousness'),
            ('extraversion', 'Extraversion'),
            ('agreeableness', 'Agreeableness'),
            ('neuroticism', 'Neuroticism'),
        ]
    )
    
    # Model parameters
    alpha = models.FloatField(default=0.001)  # L1 regularization strength
    coefficients = models.JSONField()  # Feature importance: {"0": 0.123, "1": -0.045, ...}
    intercept = models.FloatField()
    
    # Performance metrics
    train_mae = models.FloatField(null=True, blank=True)
    train_rmse = models.FloatField(null=True, blank=True)
    train_r2 = models.FloatField(null=True, blank=True)
    
    validation_mae = models.FloatField(null=True, blank=True)
    validation_rmse = models.FloatField(null=True, blank=True)
    validation_r2 = models.FloatField(null=True, blank=True)
    
    # Training data stats
    training_samples_used = models.IntegerField(default=0)
    synthetic_samples_used = models.IntegerField(default=0)
    
    # Metadata
    trained_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'lasso_model'
        unique_together = ['volunteer', 'trait']
        ordering = ['-trained_at']
    
    def __str__(self):
        return f"Lasso {self.trait.title()} Model for {self.volunteer.x_handle}"


class PSYCHOMETRIC_PROFILE(models.Model):
    """Final predicted personality profile."""
    
    id = models.BigAutoField(primary_key=True)
    volunteer = models.OneToOneField(VOLUNTEER, on_delete=models.CASCADE, related_name='psychometric_profile')
    
    # Predicted OCEAN scores (1-5 scale)
    predicted_openness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    predicted_conscientiousness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    predicted_extraversion = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    predicted_agreeableness = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    predicted_neuroticism = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1.0), MaxValueValidator(5.0)]
    )
    
    # Accuracy metrics against ground truth
    mae_openness = models.FloatField(null=True, blank=True)
    mae_conscientiousness = models.FloatField(null=True, blank=True)
    mae_extraversion = models.FloatField(null=True, blank=True)
    mae_agreeableness = models.FloatField(null=True, blank=True)
    mae_neuroticism = models.FloatField(null=True, blank=True)
    overall_mae = models.FloatField(null=True, blank=True)
    
    # Personality narrative (AI-generated)
    personality_summary = models.TextField(blank=True)
    
    # Domain recommendations (JSON)
    education_recommendations = models.JSONField(default=list)
    health_recommendations = models.JSONField(default=list)
    employment_recommendations = models.JSONField(default=list)
    ai_ethics_notes = models.TextField(blank=True)
    
    # Prediction metadata
    posts_analyzed = models.IntegerField(default=0)
    embeddings_used = models.IntegerField(default=0)
    synthetic_data_used = models.IntegerField(default=0)
    prediction_confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'psychometric_profile'
    
    def __str__(self):
        return f"Profile for {self.volunteer.x_handle}"
    
    def get_predicted_ocean_dict(self):
        """Return predicted OCEAN traits as dictionary."""
        return {
            'Openness': self.predicted_openness,
            'Conscientiousness': self.predicted_conscientiousness,
            'Extraversion': self.predicted_extraversion,
            'Agreeableness': self.predicted_agreeableness,
            'Neuroticism': self.predicted_neuroticism,
        }
    
    def get_mae_dict(self):
        """Return MAE values for each trait."""
        return {
            'Openness': self.mae_openness,
            'Conscientiousness': self.mae_conscientiousness,
            'Extraversion': self.mae_extraversion,
            'Agreeableness': self.mae_agreeableness,
            'Neuroticism': self.mae_neuroticism,
            'Overall': self.overall_mae,
        }
