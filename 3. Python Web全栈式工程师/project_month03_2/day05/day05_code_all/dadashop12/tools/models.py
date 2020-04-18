from django.db import models

class BaseModel(models.Model):
    #抽象模型类，用于继承通用字段
    created_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        #指定当前模型类 为 抽象模型类
        abstract = True


