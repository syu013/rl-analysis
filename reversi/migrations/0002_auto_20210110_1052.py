# Generated by Django 2.2.2 on 2021-01-10 01:52

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('reversi', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Trajectory',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(verbose_name='作成日時')),
                ('trajectory', models.TextField(blank=True, null=True, verbose_name='Trajectory')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to=settings.AUTH_USER_MODEL, verbose_name='ユーザー')),
            ],
            options={
                'verbose_name_plural': 'Trajectory',
            },
        ),
        migrations.DeleteModel(
            name='Reversi',
        ),
    ]
