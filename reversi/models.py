from django.db import models
from accounts.models import CustomUser

# Create your models here.
class Trajectory(models.Model):
    user = models.ForeignKey(
        CustomUser,
        verbose_name='ユーザー',
        on_delete=models.PROTECT)
    created_at = models.DateTimeField(verbose_name='作成日時')
    trajectory = models.TextField(verbose_name='Trajectory', blank=True, null=True)

    class Meta:
        verbose_name_plural = 'Trajectory'
        