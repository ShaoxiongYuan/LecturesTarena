from django.db import models

# Create your models here.

class Author(models.Model):

    name = models.CharField(max_length=11)


class Wife(models.Model):

    name = models.CharField(max_length=11)
    author = models.OneToOneField(Author, on_delete=models.CASCADE)






