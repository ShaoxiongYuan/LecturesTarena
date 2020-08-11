from django.db import models

# Create your models here.
class Author(models.Model):

    name = models.CharField(max_length=11)


class Book(models.Model):

    title = models.CharField(max_length=11)
    authors = models.ManyToManyField(Author)








