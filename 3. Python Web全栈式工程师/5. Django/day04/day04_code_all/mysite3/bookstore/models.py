from django.db import models

# Create your models here.
class Book(models.Model):

    title = models.CharField("书名",max_length=50,default='', unique=True)
    #00000.00
    price = models.DecimalField("定价", max_digits=7,decimal_places=2, default=0.0)
    #新添字段时，记住加default值
    pub = models.CharField('出版社', max_length=200, default='')
    market_price = models.DecimalField('零售价', max_digits=7, decimal_places=2, default=0.0)
    is_active = models.BooleanField('是否活跃', default=True)


    
    def __str__(self):
        return '%s_%s_%s_%s'%(self.title,self.price,self.pub,self.market_price)





class Author(models.Model):
    #表名 bookstore_author
    name = models.CharField('姓名', max_length=11, default='')
    age = models.IntegerField('年龄', default=1)
    email = models.EmailField('邮箱',null=True)




