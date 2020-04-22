# -*- coding:utf-8 -*-
from django.db import models
from utils.models import BaseModel


class UserProfile(BaseModel):

    username = models.CharField(max_length=11, verbose_name='用户名', unique=True)
    password = models.CharField(max_length=32)
    email = models.CharField(max_length=50, verbose_name='邮箱')
    phone = models.CharField(max_length=11, verbose_name='手机')
    is_active = models.BooleanField(default=False, verbose_name='激活状态')

    class Meta:
        db_table = 'user_user_profile'

    def __str__(self):
        return str(self.id)

class Address(BaseModel):
    """
    用户收货地址表：
    uid: 用户ID
    receiver: 收件人
    address: 用户地址
    default_address:是否为默认地址
    is_alive: 地址是否删除，如果地址被用户删除的话，则为False，有效的地址为True
    postcode：邮政编码
    receiver_mobile: 收件人的联系电话
    tag: 地址标签 例如：家 公司等
    """
    user_profile = models.ForeignKey('UserProfile', verbose_name='用户id',)
    receiver = models.CharField(verbose_name='收件人', max_length=10)
    address = models.CharField(max_length=100, verbose_name='用户地址')
    is_default = models.BooleanField(verbose_name='默认地址',default=False)
    # 是否用户已经删除这条地址，如果删除地址改为False，如果没删除则为True
    is_active = models.BooleanField(verbose_name='是否删除', default=True)
    postcode = models.CharField(verbose_name='邮政编码', max_length=7)
    receiver_mobile = models.CharField(verbose_name='电话', max_length=11)
    tag = models.CharField(verbose_name='标签', max_length=10, default=None)

    class Meta:
        verbose_name = '用户地址'
        verbose_name_plural = verbose_name

    def __str__(self):
        return "%s,%s,%s,%s,%s,%s" % (
        str(self.id), self.receiver, self.address, self.is_default, self.postcode, self.receiver_mobile)




class WeiboProfile(BaseModel):
    """
    用户微博表：
    {
        # 用户令牌，可以使用此作为用户的凭证
        "access_token": "2.00aJsRWFn2EsVE440573fbeaF8vtaE",
        "remind_in": "157679999",             # 过期时间
        "expires_in": 157679999,
        "uid": "5057766658",
        "isRealName": "true"
    }
    """
    user_profile = models.OneToOneField(UserProfile, null=True)
    wuid = models.CharField(verbose_name='微博uid', max_length=10, db_index=True, unique=True)
    access_token = models.CharField(verbose_name='微博授权令牌', max_length=32, db_index=True)

    class Meta:
        db_table = 'user_weibo_profile'
        verbose_name = '微博用户表'
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.wuid