from django.shortcuts import render

# Create your views here.
from django.views import View
from tools.logging_decorator import logging_check

class AdvanceOrderView(View):

    @logging_check
    def get(self,username):
        #查询字符串 -> settlement_type  0购物车
        #地址： 默认地址在数组的最前方[即0号索引位]
        #sku_list: 只显示购物车中选中的商品对应的信息
        pass