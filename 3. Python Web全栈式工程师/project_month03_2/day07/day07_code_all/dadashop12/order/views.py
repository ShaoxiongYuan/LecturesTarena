import datetime
import json

from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.
from django.views import View
from tools.logging_decorator import logging_check
from user.models import Address
from carts.views import CartsView
from django.conf import settings
from .models import OrderInfo, OrderGoods
from goods.models import SKU


class AdvanceOrderView(View):

    def get_address(self, uid):
        # 获取用户地址
        all_address = Address.objects.filter(user_profile=uid, is_active=True)

        address_default = []
        address_no_default = []
        for addr in all_address:
            addr_dict = {
                "id": addr.id,
                "name": addr.receiver,
                "mobile": addr.receiver_mobile,
                "title": addr.tag,
                "address": addr.address
            }
            if addr.is_default:
                address_default.append(addr_dict)
            else:
                address_no_default.append(addr_dict)

        return address_default + address_no_default

    def get_carts_order_list(self, uid):

        cart_obj = CartsView()
        sku_list = cart_obj.get_carts_list(uid)
        return [s for s in sku_list if s['selected'] == 1]

    @logging_check
    def get(self, request, username):
        # 查询字符串 -> settlement_type  0购物车
        # 地址： 默认地址在数组的最前方[即0号索引位]
        # sku_list: 只显示购物车中选中的商品对应的信息
        user = request.myuser
        address_list = self.get_address(user.id)
        settlement = int(request.GET['settlement_type'])

        if settlement:
            # 直接购买
            sku_list = []
        else:
            # 购物车
            sku_list = self.get_carts_order_list(user.id)
            if not sku_list:
                return JsonResponse({'code': 10500, 'error': 'The carts error'})
        data = {}
        data['addresses'] = address_list
        data['sku_list'] = sku_list
        return JsonResponse({'code': 200, 'data': data, 'base_url': settings.PIC_URL})


class OrderInfoView(View):

    def get_carts_order_data(self, uid):

        carts_obj = CartsView()
        all_data = carts_obj.get_carts_all_data(uid)
        # {skuid:[count, selected]}
        return {k: v for k, v in all_data.items() if v[1] == 1}

    def del_carts_order_data(self, uid, sku_ids):
        carts_obj = CartsView()
        for sku_id in sku_ids:
            carts_obj.del_carts_data(uid, sku_id)

    @logging_check
    def post(self, request, username):

        user = request.myuser
        json_str = request.body
        json_obj = json.loads(json_str)
        address_id = json_obj.get('address_id')
        # TODO
        # 直接购买点进来的请求 json中多两个key
        # sku_id = json_obj.get('sku_id')
        # buy_count = json_obj.get('buy_count')

        # 检查地址有效性
        try:
            address = Address.objects.get(user_profile=user.id, id=address_id)
        except Exception as e:
            return JsonResponse({'code': 10501, 'errmsg': 'The address is error'})

        # 创建订单
        with transaction.atomic():
            # 存储点
            sid = transaction.savepoint()
            now = datetime.datetime.now()
            order_id = '%s%02d' % (now.strftime('%Y%m%d%H%M%S'), user.id)
            total_count = 0
            total_amount = 0
            order = OrderInfo.objects.create(
                order_id=order_id,
                user_id=user.id,
                address=address.address,
                receiver=address.receiver,
                receiver_mobile=address.receiver_mobile,
                tag=address.tag,
                total_amount=total_amount,
                total_count=total_count,
                freight=1,
                pay_method=1,
                status=1
            )
            carts_dict = self.get_carts_order_data(user.id)
            skus = SKU.objects.filter(id__in=carts_dict.keys())
            # 检查库存 / 修改库存 / 创建订单商品数据
            for sku in skus:
                if not sku.is_launched:
                    continue
                carts_count = carts_dict[sku.id][0]
                if carts_count > sku.stock:
                    # 回滚
                    transaction.savepoint_rollback(sid)
                    return JsonResponse({'code': 10502, 'errmsg': '商品[%s]库存不足' % (sku.name)})

                old_version = sku.version
                result = SKU.objects.filter(id=sku.id, version=old_version).update(stock=sku.stock - carts_count,
                                                                                   sales=sku.sales + carts_count,
                                                                                   version=old_version + 1)
                if result == 0:
                    # 库存发生变化
                    transaction.savepoint_rollback(sid)
                    return JsonResponse({'code': 10503, 'errmsg': '服务器有点忙，请稍候重试'})
                OrderGoods.objects.create(
                    order_info_id=order_id,
                    sku_id=sku.id,
                    count=carts_count,
                    price=sku.price
                )
                # 计算订单总金额和总数量
                total_count += carts_count
                total_amount += sku.price * carts_count

            # 修改订单数据的 总金额 总数量
            order.total_amount = total_amount
            order.total_count = total_count
            order.save()

            # 提交事物
            transaction.savepoint_commit(sid)

        # TODO 购物车数据删除选中的商品
        self.del_carts_order_data(user.id, carts_dict.keys())

        data = {
            'saller': '达达商城',
            'total_amount': order.total_amount + order.freight,
            'order_freight': order.freight,
            'order_id': order_id,
            'pay_url': '',
            'carts_count': 0,  # 当前购物车key的数量
        }
        return JsonResponse({'code': 200, 'data': data})
