from django.db import models

# Create your models here.
class Book(models.Model):

    title = models.CharField("书名", max_length=11, unique=True)
    price = models.DecimalField('定价', max_digits=7, decimal_places=2)
    #新增字段 必须+default
    pub = models.CharField("出版社", max_length=20, default='')
    market_price = models.DecimalField('零售价',max_digits=6, decimal_places=2, default=0.0)
    is_active = models.BooleanField('是否活跃', default=True)


    def __str__(self):
        return 'title:%s pub:%s price:%s market_price: %s'%(self.title, self.pub, self.price, self.market_price)


    class Meta:
        db_table = 'book'


class Author(models.Model):

    name = models.CharField('姓名', max_length=11)
    age = models.IntegerField('年龄', default=1)
    email = models.EmailField('邮箱', null=True)







