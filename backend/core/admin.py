"""
Admin interface for core models.
"""
from django.contrib import admin
from django.utils.html import format_html
from .models import (
    VOLUNTEER, BFI_SURVEY, POST, BERT_EMBEDDING,
    Q_LEARNING_LOG, SYNTHETIC_DATA, LASSO_MODEL, PSYCHOMETRIC_PROFILE
)


@admin.register(VOLUNTEER)
class VolunteerAdmin(admin.ModelAdmin):
    list_display = ('x_handle', 'researcher', 'pipeline_status', 'consent_given', 'created_at')
    list_filter = ('pipeline_status', 'consent_given', 'created_at')
    search_fields = ('x_handle', 'x_user_id')
    readonly_fields = ('created_at', 'updated_at', 'posts_fetched_at')
    
    fieldsets = (
        ('Profile', {
            'fields': ('x_handle', 'x_user_id', 'researcher')
        }),
        ('Demographics', {
            'fields': ('age', 'country', 'language'),
            'classes': ('collapse',)
        }),
        ('Status', {
            'fields': ('pipeline_status', 'consent_given', 'notes')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at', 'posts_fetched_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(BFI_SURVEY)
class BFISurveyAdmin(admin.ModelAdmin):
    list_display = ('volunteer', 'openness_display', 'conscientiousness_display', 'extraversion_display', 'agreeableness_display', 'neuroticism_display', 'completed_at')
    list_filter = ('completed_at',)
    search_fields = ('volunteer__x_handle',)
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Volunteer', {
            'fields': ('volunteer',)
        }),
        ('OCEAN Scores', {
            'fields': ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')
        }),
        ('Survey Data', {
            'fields': ('responses',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('completed_at', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def openness_display(self, obj):
        if obj.openness:
            return format_html('<strong>{:.2f}</strong>', obj.openness)
        return '—'
    openness_display.short_description = 'O'
    
    def conscientiousness_display(self, obj):
        if obj.conscientiousness:
            return format_html('<strong>{:.2f}</strong>', obj.conscientiousness)
        return '—'
    conscientiousness_display.short_description = 'C'
    
    def extraversion_display(self, obj):
        if obj.extraversion:
            return format_html('<strong>{:.2f}</strong>', obj.extraversion)
        return '—'
    extraversion_display.short_description = 'E'
    
    def agreeableness_display(self, obj):
        if obj.agreeableness:
            return format_html('<strong>{:.2f}</strong>', obj.agreeableness)
        return '—'
    agreeableness_display.short_description = 'A'
    
    def neuroticism_display(self, obj):
        if obj.neuroticism:
            return format_html('<strong>{:.2f}</strong>', obj.neuroticism)
        return '—'
    neuroticism_display.short_description = 'N'


@admin.register(POST)
class PostAdmin(admin.ModelAdmin):
    list_display = ('x_post_id', 'volunteer', 'selected_by_qlearning', 'engagement_score', 'created_at_original')
    list_filter = ('selected_by_qlearning', 'embedding_processed', 'created_at_original')
    search_fields = ('x_post_id', 'volunteer__x_handle', 'content')
    readonly_fields = ('fetched_at',)
    
    fieldsets = (
        ('Post Content', {
            'fields': ('volunteer', 'x_post_id', 'content')
        }),
        ('Engagement Metrics', {
            'fields': ('retweet_count', 'like_count', 'reply_count', 'engagement_score')
        }),
        ('Q-Learning', {
            'fields': ('selected_by_qlearning', 'q_value'),
            'classes': ('collapse',)
        }),
        ('Processing', {
            'fields': ('embedding_processed', 'language_detected'),
            'classes': ('collapse',)
        }),
        ('Post Metadata', {
            'fields': ('created_at_original', 'is_reply', 'is_retweet', 'fetched_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(BERT_EMBEDDING)
class BERTEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('post', 'volunteer', 'model_name', 'created_at')
    list_filter = ('model_name', 'created_at')
    search_fields = ('post__x_post_id', 'volunteer__x_handle')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('References', {
            'fields': ('post', 'volunteer')
        }),
        ('Embedding', {
            'fields': ('model_name', 'embedding_vector'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('processing_time_seconds', 'created_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Q_LEARNING_LOG)
class QLearningLogAdmin(admin.ModelAdmin):
    list_display = ('volunteer', 'episode', 'action', 'reward', 'created_at')
    list_filter = ('action', 'episode', 'created_at')
    search_fields = ('volunteer__x_handle',)
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Episode', {
            'fields': ('volunteer', 'episode', 'post')
        }),
        ('Decision', {
            'fields': ('action', 'state', 'reward')
        }),
        ('Q-Learning Update', {
            'fields': ('old_q_value', 'new_q_value', 'learning_rate', 'discount_factor'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(SYNTHETIC_DATA)
class SyntheticDataAdmin(admin.ModelAdmin):
    list_display = ('volunteer', 'original_post', 'generation_confidence', 'used_in_training', 'created_at')
    list_filter = ('generator_version', 'used_in_training', 'created_at')
    search_fields = ('volunteer__x_handle',)
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Reference', {
            'fields': ('volunteer', 'original_post', 'original_embedding')
        }),
        ('Generated Data', {
            'fields': ('synthetic_text', 'synthetic_embedding'),
            'classes': ('collapse',)
        }),
        ('GAN Info', {
            'fields': ('generator_version', 'generation_confidence', 'used_in_training')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(LASSO_MODEL)
class LassoModelAdmin(admin.ModelAdmin):
    list_display = ('volunteer', 'trait', 'is_active', 'train_mae', 'validation_mae', 'trained_at')
    list_filter = ('trait', 'is_active', 'trained_at')
    search_fields = ('volunteer__x_handle',)
    readonly_fields = ('trained_at', 'updated_at')
    
    fieldsets = (
        ('Model', {
            'fields': ('volunteer', 'trait', 'is_active')
        }),
        ('Parameters', {
            'fields': ('alpha', 'intercept'),
            'classes': ('collapse',)
        }),
        ('Training Performance', {
            'fields': ('train_mae', 'train_rmse', 'train_r2', 'training_samples_used')
        }),
        ('Validation Performance', {
            'fields': ('validation_mae', 'validation_rmse', 'validation_r2'),
            'classes': ('collapse',)
        }),
        ('Coefficients', {
            'fields': ('coefficients',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('trained_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(PSYCHOMETRIC_PROFILE)
class PsychometricProfileAdmin(admin.ModelAdmin):
    list_display = ('volunteer', 'overall_mae', 'prediction_confidence', 'posts_analyzed', 'updated_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('volunteer__x_handle',)
    readonly_fields = ('created_at', 'updated_at')
    
    fieldsets = (
        ('Volunteer', {
            'fields': ('volunteer',)
        }),
        ('Predicted OCEAN Scores', {
            'fields': ('predicted_openness', 'predicted_conscientiousness', 'predicted_extraversion', 'predicted_agreeableness', 'predicted_neuroticism')
        }),
        ('Accuracy (MAE)', {
            'fields': ('mae_openness', 'mae_conscientiousness', 'mae_extraversion', 'mae_agreeableness', 'mae_neuroticism', 'overall_mae'),
            'classes': ('collapse',)
        }),
        ('Domain Recommendations', {
            'fields': ('education_recommendations', 'health_recommendations', 'employment_recommendations', 'ai_ethics_notes'),
            'classes': ('collapse',)
        }),
        ('Summary', {
            'fields': ('personality_summary',),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('posts_analyzed', 'embeddings_used', 'synthetic_data_used', 'prediction_confidence', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


# Customize Django admin site
admin.site.site_header = "Big Five Personality Prediction System"
admin.site.site_title = "Admin Portal"
admin.site.index_title = "Dashboard"
