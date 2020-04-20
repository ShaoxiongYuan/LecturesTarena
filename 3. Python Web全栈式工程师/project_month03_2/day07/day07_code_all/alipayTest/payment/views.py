from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views import View
import json
from alipay import AliPay
from django.conf import settings

app_private_key_string = open(settings.ALIPAY_KEY_DIRS + 'app_private_key.pem').read()
alipay_public_key_string = open(settings.ALIPAY_KEY_DIRS + 'alipay_public_key.pem').read()

ORDER_STATUS = 1


class MyAliPay(View):
    # 自定义视图类的基类
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alipay = AliPay(
            appid=settings.ALIPAY_APP_ID,
            app_notify_url=None,
            # 应用私钥
            app_private_key_string=app_private_key_string,
            # 支付宝公钥
            alipay_public_key_string=alipay_public_key_string,
            # 指明签名算法
            sign_type="RSA2",
            # debug模式 会把请求发送至沙箱环境
            debug=True
        )

    def get_trade_url(self, order_id, amount):
        # pc网站支付地址  https://openapi.alipaydev.com/gateway.do? + querystring
        order_string = self.alipay.api_alipay_trade_page_pay(
            # 订单号
            out_trade_no=order_id,
            # 订单总金额
            total_amount=amount,
            subject=order_id,
            return_url=settings.ALIPAY_RETURN_URL,
            notify_url=settings.ALIPAY_NOTIFY_URL
        )

        return "https://openapi.alipaydev.com/gateway.do?" + order_string

    def get_verify_result(self, data, sign):
        # 验签成功 返回 True   失败是 False
        return self.alipay.verify(data, sign)

    def get_trade_result(self, order_id):
        # 查询支付结果
        result = self.alipay.api_alipay_trade_query(out_trade_no=order_id)
        print(result)
        if result.get("trade_status") == "TRADE_SUCCESS":
            return True
        return False


# Create your views here.
class OrderInfoView(MyAliPay):

    def get(self, request):
        return render(request, 'alipay.html')

    def post(self, request):
        json_obj = json.loads(request.body)
        order_id = json_obj.get('order_id')
        pay_url = self.get_trade_url(order_id, 999)
        return JsonResponse({'code': 200, 'pay_url': pay_url})


class ResultView(MyAliPay):

    def get(self, request):
        print(request.GET)
        request_data = {k: request.GET[k] for k in request.GET.keys()}
        sign = request_data.pop('sign')
        is_verify = self.get_verify_result(request_data, sign)
        if is_verify:
            # 消息来源可靠
            order_id = request_data.get('out_trade_no')
            # 去自己数据库中查询该订单的状态;例如 dashop12中订单状态案例，此时订单状态应该进入到 ‘待发货/已付款’
            if ORDER_STATUS == 2:
                return HttpResponse('支付成功')
            else:
                # 触发主动查询
                res = self.get_trade_result(order_id)
                if res:
                    # 支付是成功的，更改订单状态
                    return HttpResponse('主动查询得知支付成功')
                else:
                    return HttpResponse('主动查询得知支付未成功')
        else:
            return HttpResponse("非法访问")

    def post(self, request):
        # notify_url 的处理，支付宝发表单post请求，将支付结果告知 商户
        request_data = {k: request.POST[k] for k in request.POST.keys()}
        sign = request_data.pop('sign')
        is_verify = self.get_verify_result(request_data, sign)

        if is_verify:
            trade_status = request_data.get('trade_status')
            if trade_status == 'TRADE_SUCCESS':
                # 更改订单状态 用户支付成功
                return HttpResponse("success")
        else:
            return HttpResponse('非法访问')
