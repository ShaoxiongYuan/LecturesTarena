from django.db import models


# Create your models here.
class Author(models.Model):
    name = models.CharField(max_length=11, verbose_name='作家名称')


class Wife(models.Model):
    name = models.CharField(max_length=11, verbose_name='妻子名称')
    author = models.OneToOneField(Author)
