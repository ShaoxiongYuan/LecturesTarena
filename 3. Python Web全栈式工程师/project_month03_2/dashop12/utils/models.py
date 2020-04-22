from django.db import models


class BaseModel(models.Model):
    """为模型类补充字段"""
    # 此为抽象类,用于继承创建时间和修改时间字段
    created_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        abstract = True