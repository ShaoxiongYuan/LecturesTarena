from django.views import View
import json
from goods.models import *
# Create your views here.
from django.http import JsonResponse
from django_redis import get_redis_connection
from utils.loging_decorator import logging_check
#from dadashop.settings import PIC_URL
from django.conf import settings
from goods.views import get_sku_name_and_values

redis_conn = get_redis_connection('carts')


# 购物车
class CartsView(View):
    @logging_check
    def dispatch(self, request, *args, **kwargs):
        json_str = request.body
        # 小心判断 是否存在，get时没有json_str
        request.json_obj = {}
        if json_str:
            json_obj = json.loads(json_str)
            request.json_obj = json_obj
        # 尝试添加sku
        request.sku = None
        if 'sku_id' in request.json_obj:
            try:
                sku = SKU.objects.get(id=request.json_obj['sku_id'])
            except Exception as e:
                return None
            request.sku = sku
        return super().dispatch(request, *args, **kwargs)

    def del_carts_data(self, user_id, sku_id):
        '''

        :param user_id:  用户id ：user.id
        :param sku_id:   SKU表商品id : sku_id
        :return:
        '''
        redis_conn.hdel('carts_%d' % user_id, sku_id)



    def get_cache_key(self, user_id):
        '''

        :param user_id:  user.id 用户的id
        :return: redis的键名
        '''
        return 'carts_%s' % (user_id)

    def get_carts_data(self, user_id, sku_id):
        '''

        :param user_id:  user.id 用户的id
        :param sku_id:  sku_id  SKU表商品的id
        :return:
        '''
        key = self.get_cache_key(user_id)
        value = redis_conn.hget(key, sku_id)
        if not value:
            return None
        return json.loads(value)

    def get_carts_datas(self, user_id):
        '''

        :param user_id:  user.id 用户的id
        :return: data = redis存的键和值 键是：字节串 user_id ，值是：字节串 {'count':xx,'selected':xxx}
        '''
        key = self.get_cache_key(user_id)
        o_data = redis_conn.hgetall(key)
        if not o_data:
            return {}
        # {b'1':b"{}"}
        data = {int(k): json.loads(v) for k, v in o_data.items()}
        return data

    def set_carts_data(self, user_id, sku_id, data):
        '''

        :param user_id: user.id 用户id
        :param sku_id: sku.id sku表id
        :param data: data 是redis存的字典数据 是：字节串 {'count':xx,'selected':xxx}
        :return:
        '''
        data_s = json.dumps(data)
        key = self.get_cache_key(user_id)
        redis_conn.hset(key, sku_id, data_s)

    def set_select_unselect(self, user_id, sku_id, selected):
        '''
        put 请求的state 状态 select 和  unselect
        :param user_id: 用户id user.id
        :param sku_id: 商品id int sku_id
        :param selected: 状态码（0/1)
        :return: 响应列表 [{"id":"","name":"","count":"","selected":""},{"":""...}]
        '''
        carts = self.get_carts_data(user_id, sku_id)
        if not carts:
            return None

        info = {'count': carts['count'], 'selected': selected}
        self.set_carts_data(user_id, sku_id, info)
        skus_list = self.get_carts_list(user_id)
        return skus_list

    def set_selectall_unselectall(self, user_id, selected):
        '''
        put 请求的state 状态selectall 和 unselectall
        :param user_id: 用户id user.id
        :param selected: 状态码（0/1)
        :return: 响应列表 [{"id":"","name":"","count":"","selected":""},{"":""...}]
        '''

        carts = self.get_carts_datas(user_id)
        if not carts:
            return None
        for sku_id, data in carts.items():
            info = {'count': data['count'], 'selected': selected}
            self.set_carts_data(user_id, sku_id, info)

        skus_list = self.get_carts_list(user_id)
        return skus_list

    def get_carts_list(self, user_id):
        '''

        :param user_id: 用户的id
        :return: reqsponse [get,put]列表 [{"id":"","name":"","count":"","selected":"","default_image_url":"","price":"","sku_sale_attr_name":[],"sku_sale_attr_val":[]},{"":""...}]
        '''

        carts_dict = self.get_carts_datas(user_id)
        if not carts_dict:
            return []

        skus = SKU.objects.filter(id__in=carts_dict.keys())
        skus_list = []
        for sku in skus:
            sku_dict = {}
            sku_dict['id'] = sku.id
            sku_dict['name'] = sku.name
            sku_dict['count'] = int(carts_dict[sku.id]['count'])
            sku_dict['selected'] = int(carts_dict[sku.id]['selected'])
            sku_dict['default_image_url'] = str(sku.default_image_url)
            sku_dict['price'] = sku.price

            sku_sale_attr_val, sku_sale_attr_name = get_sku_name_and_values(sku.id)

            sku_dict['sku_sale_attr_name'] = sku_sale_attr_name
            sku_dict['sku_sale_attr_val'] = sku_sale_attr_val
            skus_list.append(sku_dict)
        return skus_list



    def post(self, request, username):
        count = request.json_obj['count']
        uid = request.user.id
        sku = request.sku

        if not sku:
            return JsonResponse({'code': 30102, 'error': '没有sku参数'})

        if request.user.username != username:
            return JsonResponse({'code': 30104, 'error': '非登录者用户'})

        try:
            count = int(count)
        except Exception as e:
            print(e)
            return JsonResponse({'code': 30102, 'error': "传参不正确"})

        # 判断库存
        if count > sku.stock:
            return JsonResponse({'code': 9999, 'error': '超量'})

        carts = self.get_carts_datas(uid)
        if not carts:
            # 默认选中
            info = {'count': count, 'selected': 1}
            self.set_carts_data(uid, sku.id, info)
            count = len(carts)
            return JsonResponse({'code': 200, 'data': {'carts_count': count}, 'base_url': settings.PIC_URL})

        else:
            my_sku_info = carts.get(sku.id)
            if not my_sku_info:
                my_sku_info = {'count': count, 'selected': 1}
            else:
                old_count = my_sku_info['count']
                new_count = old_count + count
                if new_count > sku.stock:
                    return JsonResponse({'code': 9999, 'error': '超量'})
                my_sku_info['count'] = new_count

            self.set_carts_data(uid, sku.id, my_sku_info)
            carts_data = self.get_carts_datas(uid)
            count = len(carts_data)
            return JsonResponse({'code': 200, 'data': {'carts_count': count}, 'base_url': settings.PIC_URL})

    # 查询购物车
    def get(self, request, username):
        user = request.user
        if user.username != username:
            return JsonResponse({'code': 30104, 'error': '非登录者用户'})
        skus_list = self.get_carts_list(user.id)
        return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})

    # 删除购物车
    def delete(self, request, username):
        uid = request.user.id
        sku = request.sku
        if not sku:
            return JsonResponse({'code': 30102, 'error': '没有sku参数'})
        user = request.user
        if user.username != username:
            return JsonResponse({'code': 30104, 'error': '非登录者用户'})

        # 从hash值中删除该SKU_ID
        self.del_carts_data(uid, sku.id)
        carts_data = self.get_carts_datas(uid)
        count = len(carts_data)
        return JsonResponse({'code': 200, 'data': {'carts_count': count}, 'base_url': settings.PIC_URL})

    def put(self, request, username):
        state = request.json_obj['state']
        user = request.user
        if user.username != username:
            return JsonResponse({'code': 30104, 'error': '非登录者用户'})

        # 判断增加还是减少
        if state == 'add' or state == "del":
            sku = request.sku
            if not sku:
                return JsonResponse({'code': 30102, 'error': '没有sku参数'})
            carts = self.get_carts_data(user.id, sku.id)
            if not carts:
                return JsonResponse({'code': 30101, 'error': '未找到商品'})
            count = carts['count']

            # 检查数据
            if state == 'add':
                # 向hash中存储商品的ID,和数量
                count += 1
                if count > sku.stock:
                    return JsonResponse({'code': 30103, 'error': '购买数量超过库存'})
                info = {'count': count, 'selected': 1}
                self.set_carts_data(user.id, sku.id, info)
                skus_list = self.get_carts_list(user.id)
                if not skus_list:
                    return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})

            elif state == 'del':
                if count > 1:
                    count -= 1
                    info = {'count': count, 'selected': 1}
                    self.set_carts_data(user.id, sku.id, info)
                    skus_list = self.get_carts_list(user.id)
                else:
                    info = {'count': 1, 'selected': 1}
                    self.set_carts_data(user.id, sku.id, info)
                    skus_list = self.get_carts_list(user.id)
                    if not skus_list:
                        return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})
        # 判断是否勾选
        if state in ('select', 'unselect'):
            sku = request.sku
            if not sku:
                return JsonResponse({'code': 30102, 'error': '没有sku参数'})
            # 勾选
            if state == 'select':
                skus_list = self.set_select_unselect(user.id, sku.id, 1)
                if not skus_list:
                    return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})

            # 取消勾选
            if state == 'unselect':
                skus_list = self.set_select_unselect(user.id, sku.id, 0)
                if not skus_list:
                    return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})

        # 判断是否全选
        if state in ('selectall', 'unselectall'):
            if state == 'selectall':
                skus_list = self.set_selectall_unselectall(user.id, 1)
                if not skus_list:
                    return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})

            if state == 'unselectall':
                skus_list = self.set_selectall_unselectall(user.id, 0)
                if not skus_list:
                    return JsonResponse({'code': 30101, 'error': '未找到商品'})
                return JsonResponse({'code': 200, 'data': skus_list, 'base_url': settings.PIC_URL})



    def merge_carts(self, user_id, carts_info):
        # 合并购物车
        # 返回购物车物品数量
        carts_data = self.get_carts_datas(user_id)

        if not carts_info:
            # 如果用户未登录时购物车中无数据
            return len(carts_data)

        for c_dic in carts_info:
            sku_id = int(c_dic['id'])
            sku_data = SKU.objects.get(id=sku_id)
            c_count = int(c_dic['count'])

            if sku_id in carts_data:
                sku_count = carts_data[sku_id]['count']
                last_count = min(sku_data.stock, max(sku_count, c_count))
                carts_data[sku_id]['count'] = last_count
                # 合并selected状态
            else:
                carts_data[sku_id] = {'count': min(c_count, sku_data.stock), 'selected': 1}
            self.set_carts_data(user_id, sku_id, carts_data[sku_id])

        return len(carts_data)
