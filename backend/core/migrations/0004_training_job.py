from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('core', '0003_lasso_model_classification_accuracy_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='TRAINING_JOB',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('job_type', models.CharField(default='pandora_training', max_length=40)),
                ('task_id', models.CharField(blank=True, default='', max_length=255)),
                ('status', models.CharField(choices=[('queued', 'Queued'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed')], db_index=True, default='queued', max_length=20)),
                ('stage', models.CharField(default='queued', max_length=80)),
                ('progress', models.IntegerField(default=0)),
                ('message', models.TextField(blank=True, default='')),
                ('error', models.TextField(blank=True, default='')),
                ('result', models.JSONField(blank=True, default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('started_at', models.DateTimeField(blank=True, null=True)),
                ('finished_at', models.DateTimeField(blank=True, null=True)),
                ('researcher', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='training_jobs', to=settings.AUTH_USER_MODEL)),
                ('volunteer', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='training_jobs', to='core.volunteer')),
            ],
            options={
                'db_table': 'training_job',
                'ordering': ['-created_at'],
            },
        ),
    ]
