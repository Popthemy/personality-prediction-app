from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='COHORT_MODEL',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=120, unique=True)),
                ('version', models.CharField(default='v1', max_length=40)),
                ('is_active', models.BooleanField(db_index=True, default=True)),
                ('split_seed', models.IntegerField(default=42)),
                ('train_ratio', models.FloatField(default=0.8)),
                ('validation_ratio', models.FloatField(default=0.2)),
                ('train_volunteer_ids', models.JSONField(default=list)),
                ('validation_volunteer_ids', models.JSONField(default=list)),
                ('train_handles', models.JSONField(default=list)),
                ('validation_handles', models.JSONField(default=list)),
                ('trainer_state', models.JSONField(default=dict)),
                ('metrics', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'cohort_model',
                'ordering': ['-updated_at'],
            },
        ),
    ]
