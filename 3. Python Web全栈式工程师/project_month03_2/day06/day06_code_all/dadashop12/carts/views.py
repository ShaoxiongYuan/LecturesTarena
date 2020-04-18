import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from tools.logging_decorator import logging_check
from goods.models import SKU
from django.core.cache import caches
from django.conf import settings

CARTS_CACHE = caches['carts']


# Create your views here.
class CartsView(View):


    def get_cache_key(self, uid):
        #获取购物车在redis中的key
        return 'carts_%s'%(uid)


    def get_carts_all_data(self, uid):
        #获取用户购物车数据
        # carts_1 : {sku_id:[count,selected]}
        # selected 为选中状态 0 未选中  1 选中
        key = self.get_cache_key(uid)
        data = CARTS_CACHE.get(key)
        if not data:
            return {}
        r_data = {int(k):v for k, v in data.items()}
        return r_data


    def set_carts_data(self, uid, sku_id, data):
        #存储购物车数据
        key = self.get_cache_key(uid)
        all_data = self.get_carts_all_data(uid)
        all_data[int(sku_id)] = data
        CARTS_CACHE.set(key, all_data)

    def get_carts_list(self, uid):
        #生成前端购物车数据结构

        carts_data = self.get_carts_all_data(uid)
        if not carts_data:
            return []

        skus = SKU.objects.filter(id__in=carts_data.keys())
        skus_list = []

        # [{"id":"","name":"","count":"","selected":"","default_image_url":"","price":"","sku_sale_attr_name":[],"sku_sale_attr_val":[]},{"":""...}]
        for sku in skus:
            sku_dict = {}
            sku_dict['id'] = sku.id
            sku_dict['name'] = sku.name
            sku_dict['count'] = carts_data[sku.id][0]
            sku_dict['selected'] = carts_data[sku.id][1]
            sku_dict['price'] = str(sku.price)
            sku_dict['default_image_url'] = str(sku.default_image_url)
            sku_sale_attr_name = []
            sku_sale_attr_value= []
            #sku正向查询，查询出对应的sale_attr_value
            sale_attr_values = sku.sale_attr_value.all()
            for attr_vaule in sale_attr_values:
                sku_sale_attr_value.append(attr_vaule.name)
                sku_sale_attr_name.append(attr_vaule.spu_sale_attr.name)
            sku_dict['sku_sale_attr_name'] = sku_sale_attr_name
            sku_dict['sku_sale_attr_val'] = sku_sale_attr_value
            skus_list.append(sku_dict)

        return skus_list

    def merge_carts(self, uid, carts_info):
        #合并前后端购物车
        carts_data = self.get_carts_all_data(uid)

        if not carts_info:
            #用户离线状态下 未使用购物车
            return len(carts_data)

        for c_dic in carts_info:
            sku_id = int(c_dic['id'])
            try:
                sku_data = SKU.objects.get(id=sku_id, is_launched=True)
            except Exception as e:
                continue
            c_count = int(c_dic['count'])

            #判断后端购物车是否有该商品
            if sku_id in carts_data:
                #后端购物车有该商品
                sku_count = carts_data[sku_id][0]
                last_count =min(sku_data.stock, max(sku_count, c_count))
                carts_data[sku_id][0] = last_count
            else:
                #后端购物车没有该商品
                carts_data[sku_id] = [min(sku_data.stock, c_count), 1]
            self.set_carts_data(uid, sku_id, carts_data[sku_id])

        return len(carts_data)



    @logging_check
    def get(self, request, username):

        user = request.myuser
        skus_list = self.get_carts_list(user.id)

        return JsonResponse({'code':200, 'data':skus_list, 'base_url': settings.PIC_URL})

    @logging_check
    def post(self, request, username):

        json_str = request.body
        json_obj = json.loads(json_str)
        sku_id = json_obj['sku_id']
        count = json_obj['count']

        #检查sku的合法性
        try:
            sku = SKU.objects.get(id=sku_id, is_launched=True)
        except Exception as e:
            return JsonResponse({'code':10400, 'error':'The sku_id is wrong !'})

        count = int(count)
        #判断库存是否合法
        if count > sku.stock:
            return JsonResponse({'code':10401, 'error': 'The count is error !'})

        #检查购物车数据
        user = request.myuser
        carts = self.get_carts_all_data(user.id)
        if not carts:
            #第一次使用购物车功能
            my_sku_info = [count, 1]
        else:
            my_sku_info = carts.get(sku.id)
            if not my_sku_info:
                #购物之前没有添加过该商品
                my_sku_info = [count, 1]
            else:
                #购物车当前已经有这个商品
                old_count = my_sku_info[0]
                new_count = old_count + count
                if new_count > sku.stock:
                    return JsonResponse({'code':10402, 'error':'The total count is error !'})
                my_sku_info[0] = new_count

        #存储购物车数据
        self.set_carts_data(user.id, sku.id, my_sku_info)
        carts_data = self.get_carts_all_data(user.id)
        sku_count = len(carts_data)
        return JsonResponse({'code':200, 'data':{'carts_count':sku_count},'base_url':settings.PIC_URL})















        return JsonResponse({'code':200})