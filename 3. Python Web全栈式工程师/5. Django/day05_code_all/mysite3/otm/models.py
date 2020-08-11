from django.db import models

# Create your models here.
class Publisher(models.Model):
    #出版社 [一]
    name = models.CharField(max_length=11)


class Book(models.Model):
    #书 [多]
    title = models.CharField(max_length=11)
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)






