"""
Custom user model for researcher accounts.
"""
from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """Extended user model for researchers."""
    
    user_type = models.CharField(
        max_length=20,
        choices=[
            ('researcher', 'Researcher'),
            ('admin', 'Administrator'),
        ],
        default='researcher'
    )
    
    institution = models.CharField(max_length=255, blank=True)
    bio = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to='profiles/', null=True, blank=True)
    
    # Research settings
    max_volunteers = models.IntegerField(default=100)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'custom_user'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.get_full_name() or self.username
