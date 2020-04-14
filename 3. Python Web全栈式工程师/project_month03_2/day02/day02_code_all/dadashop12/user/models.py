from django.db import models

# Create your models here.
class UserProfile(models.Model):

    username = models.CharField(max_length=11, verbose_name='用户名', unique=True)
    password = models.CharField(max_length=32)
    email = models.EmailField()
    phone = models.CharField(max_length=11)
    is_active = models.BooleanField(default=False, verbose_name='激活状态')

    class Meta:
        db_table = 'user_user_profile'

    def __str__(self):
        return '%s_%s'%(self.id,self.username)





