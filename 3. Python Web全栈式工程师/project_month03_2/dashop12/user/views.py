# Create your views here.
import hashlib
import random
import base64
import json

from django.views.generic import View
from django.http import JsonResponse
from django.db import transaction
from django.conf import settings
from dtoken.views import make_token
from .weiboapi import OAuthWeibo
from utils.loging_decorator import logging_check
from .tasks import send_active_email, send_password_email
from .models import UserProfile, Address, WeiboProfile
from carts.views import CartsView
from django.core.cache import caches

EMAIL_CACHE = caches['verify_email']

# Create your views here.

class ModifyPasswordView(View):
    """
    用户登陆状态下 修改密码：
    http://127.0.0.1:8000/v1/user/<username>/password
    """
    @logging_check
    def post(self, request, username):
        """
        :param request: 请求对象
        :return: 返回修改密码之后的状态
        """
        data = json.loads(request.body)
        oldpassword = data.get('oldpassword')
        password1 = data.get('password1')
        password2 = data.get('password2')
        if oldpassword == password1 or oldpassword == password2:
            return JsonResponse({'code': 10109, 'error': 'Please Use Different Password!'})
        # 判断两次密码是否一致
        if password1 != password2:
            return JsonResponse({'code': 10102, 'error': 'Inconsistent passwords!'})
        user = request.user
        real_password = user.password
        m = hashlib.md5()
        m.update(oldpassword.encode())
        if m.hexdigest() != real_password:
            return JsonResponse({'code': 10103, 'error': 'Old password error!'})
        new = hashlib.md5()
        new.update(password1.encode())
        user.password = new.hexdigest()
        user.save()
        return JsonResponse({'code': 200, 'data': 'OK'})


class SendSmsCodeView(View):
    """
    用户找回密码视图处理函数：
    分为三步：
    1.验证邮箱，并且发送邮件验证码
    2.验证邮件验证码，
    3.验证码验证成功，修改密码
    """
    def post(self, request):
        data = json.loads(request.body)

        # 验证用户是否是已经注册用户
        email = data.get('email')
        try:
            user = UserProfile.objects.get(email=email)
        except Exception as e:
            return JsonResponse({'code': 10104, 'error':'User query error'})
        # 先去查询该用户是否在时效时间内发送过验证码

        try:
            email_code = EMAIL_CACHE.get('email_code_%s'%email)
        except Exception as e:
            return JsonResponse({'code': 10132, 'error': 'Verify Code Error'})
        if email_code:
            return JsonResponse({'code': 202, 'error': 'please enter your code!'})

        email_code = "%06d" % random.randint(0, 999999)
        try:
            EMAIL_CACHE.set("email_code_%s" % email, email_code,10 * 60)
        except Exception as e:
            return JsonResponse({'code': 10105, 'error': 'Storage authentication code failed'})
        send_password_email.delay(email, email_code)
        return JsonResponse({'code': 200, 'data':'OK'})


class VerifyCodeView(View):
    """
    第二步 验证发送邮箱的验证码
    """
    def post(self, request):
        """
        验证用户邮箱验证码
        :param request:
        :param username: 用户名
        :return:
        """
        data = json.loads(request.body)
        email = data.get('email')
        code = data.get('code')
        # 验证用户是否匹配
        try:
            email_code = EMAIL_CACHE.get('email_code_%s' % email)
        except Exception as e:
            return JsonResponse({'code': 10106,'error': 'invalid validation. Resend it.'})
        redis_code = email_code.decode()
        if redis_code == code:
            return JsonResponse({'code': 200, 'data': '验证码通过！', 'email': email})
        return JsonResponse({'code': 10124, 'error':'验证码错误！'})
        

class ModifyPwdView(View):
    """
    最后一步验证邮箱，修改密码
    """

    def post(self, request):
        data = json.loads(request.body)
        password1 = data.get('password1')
        password2 = data.get('password2')
        email = data.get('email')
        if password1 != password2:
            return JsonResponse({'code': 10110, 'error': 'Password Inconsistencies!'})
        try:
            user = UserProfile.objects.get(email=email)
        except Exception as e:
            return JsonResponse({'code': 10104, 'error': 'User query error!'})
        # 读取旧密码
        new = hashlib.md5()
        new.update(password1.encode())
        user.password = new.hexdigest()
        user.save()
        return JsonResponse({'code': 200, 'data': 'OK'})


class ActiveView(View):
    """
    # 用户发送邮件激活
    # GET http://127.0.0.1:8000/v1/user/active?code=xxxxx
    """
    def get(self, request):
        """
        由于没有设置激活链接的参数的redis中的有效时间。
        在用户激活之后删除redis中缓存的激活链接
        """
        code = request.GET.get('code', None)
        if not code:
            return JsonResponse({'code': 10113, 'error':'Error activating link parameters'})
        # 反解激活验证链接
        verify_code = base64.urlsafe_b64decode(code.encode()).decode()
        random_code, username = verify_code.split('/')
        result = EMAIL_CACHE.get('email_active_%s' % username)
        if not result:
            return JsonResponse({'code': 10112, 'error': 'Link is invalid and resend it!'})
        # 验证前端传来的激活链接和redis中是否一致
        if random_code != result:
            return JsonResponse({'code': 10112, 'error': 'Link is invalid and resend it!'})
        # 判断code中的用户信息和数据库中信息是否一致
        try:
            user = UserProfile.objects.get(username=username)
        except Exception as e:
            return JsonResponse({'code': 10122, 'error':'User query error'})
        user.is_active = True
        user.save()
        EMAIL_CACHE.delete('email_active_%s'%username)
        return JsonResponse({'code': 200, 'data': 'OK'})


class AddressView(View):
    """
    get: 获取用户的绑定的收货地址
    post: 新增用户绑定的收货地址
    delete：实现用户删除地址功能
    put: 实现用户修改地址功能
    """
    @logging_check
    def get(self, request, username):
        """
        返回用户关联的地址页面，以及地址
        :param request:
        :return: addressAdmin.html & addresslist
        """
        user = request.user
        try:
            all_address = Address.objects.filter(user_profile=user, is_active=True)
        except Address.DoesNotExist as e:
            return JsonResponse({'code': 10114, 'error': 'Error in Address Query!'})
        addresslist = []
        for values in all_address:
            each_address = {}
            each_address['id'] = values.id
            each_address['address'] = values.address
            each_address['receiver'] = values.receiver
            each_address['receiver_mobile'] = values.receiver_mobile
            each_address['tag'] = values.tag
            each_address['is_default'] = values.is_default
            addresslist.append(each_address)
        return JsonResponse({'code': 200,'addresslist': addresslist})

    @logging_check
    def post(self, request, username):
        """
        用来提交保存用户的收获地址
        1.先获取相应的用户，然后根据用户的id来绑定地址
        :param request:
        :return:返回保存后的地址以及地址的id
        """
        data = json.loads(request.body)
        receiver = data.get('receiver')
        address = data.get('address')
        receiver_phone = data.get('receiver_phone')
        postcode = data.get('postcode')
        tag = data.get('tag')
        user = request.user
        # 先查询当前用户有没有保存的地址。
        # 如果有则需要将default_address 设置为False
        # 如果没有则需要default_address 设置为True
        query_address = Address.objects.filter(user_profile=user)
        default_status = False
        if not query_address:
            default_status = True
        try:
            Address.objects.create(
                user_profile=user,
                receiver=receiver,
                address=address,
                is_default=default_status,
                receiver_mobile=receiver_phone,
                is_active=True,
                postcode=postcode,
                tag=tag,
            )
        except Exception as e:
            return JsonResponse({'code': 10120, 'error': 'Address storage exception'})
        return JsonResponse({'code':200,'data':'新增地址成功！'})

    @logging_check
    def delete(self, request, username, id):
        """
         删除用户的提交的地址
         :param request: 提交的body中为用户的地址的id
         :param username:
         :return: 删除后用户的所有的收货地址
        """
        # 根据用户发来的地址的id来直接删除用户地址
        if not id:
            return JsonResponse({'code': 10122, 'error': 'Get address id error'})
        try:
            address = Address.objects.get(id=id)
        except Address.DoesNotExist as e:
            # 此刻应该写个日志
            return JsonResponse({'code': 10121, 'error': 'Get address exception'})
        #默认地址不让删除
        if address.is_default:
            return JsonResponse({'code': 10121, 'error': 'The defualt address can not delete'})

        try:
            address.is_active = False
            address.save()
        except Exception as e:
            return JsonResponse({'code':10122,'error':'delete address error'})
        return JsonResponse({'code':200, 'data':'删除地址成功！'})
        
    @logging_check
    def put(self, request, username, id):
        """
        根据用户传递过来的收货地址来修改相应的内容
        :param request: 用户请求的对象
        :param address_id:用户地址id
        :return: 返回修改之后的地址的全部内容
        """
        if not id:
            return JsonResponse({'code': 10122, 'error': 'Get address id error'})
        data = json.loads(request.body)
        address = data.get('address')
        receiver = data.get('receiver')
        tag = data.get('tag')
        receiver_mobile = data.get('receiver_mobile')
        data_id = data.get('id')
        if int(id) != data_id:
            return JsonResponse({'code':12345,'error':'ID error'})
        # 1  根据地址的id筛选出那一条记录
        try:
            filter_address = Address.objects.get(id=id)
        except Exception as e:
            return JsonResponse({'code': 10122, 'error': 'Get address exception!'})
        try:        
            filter_address.receiver = receiver
            filter_address.receiver_mobile =receiver_mobile
            filter_address.address = address
            filter_address.tag = tag
            filter_address.save()
        except Exception as e:
            return JsonResponse({'code':10123,'error':'修改地址失败！'})
        return JsonResponse({'code':200, 'data':'修改地址成功！'})


class DefaultAddressView(View):
    """
    用来修改默认地址
    """
    @logging_check
    def post(self, request, username):
        """
        用来修改默认地址
        :param address_id:用户修改地址的id
        """
        # 先根据address_id 来匹配出用户的id
        data = json.loads(request.body)
        address_id = data.get('id')
        # 需要将此条地址设为默认地址，其他的需要设置为非默认地址。
        user = request.user
        #事务操作
        with transaction.atomic():
            old_defaults = Address.objects.filter(user_profile=user, is_active=True, is_default=True)
            if old_defaults:
                old_default = old_defaults[0]
                old_default.is_default = False
                old_default.save()
            new_defaults = Address.objects.filter(id=address_id,is_active=True)
            if new_defaults:
                new_default = new_defaults[0]
                new_default.is_default = True
                new_default.save()

        return JsonResponse({'code':200,'data':'设为默认成功！'})


class OAuthWeiboUrlView(View):
    def get(self, request):
        """
        用来获取微博第三方登陆的url,返回值为微博登陆页的地址。
        """
        try:
            oauth_weibo = OAuthWeibo()
            oauth_weibo_url = oauth_weibo.get_weibo_login_url()
        except Exception as e:
            print(e)
            return JsonResponse({'code': 10124, 'error': 'Cant get weibo login page'})
        return JsonResponse({'code': 200, 'oauth_url': oauth_weibo_url})


class OAuthWeiboView(View):
    def get(self, request):
        """
        获取用户的code,以及用户的token
        """
        # 首先获取两个参数code 和state
        code = request.GET.get('code', None)
        if not code:
            return JsonResponse({'code': 10100, 'error':  'Invalid parameters'})
        try:
            oauth_weibo = OAuthWeibo()
            userInfo = oauth_weibo.get_access_token(code)
        except Exception as e:
            return JsonResponse({'code':10142,'error':'cant reach weibo server'})
        # 将用户weibo的uid传入到前端
        # OAuth 2.0 中授权码模式下 如果错误的请求，响应中会字典中会有error键
        if userInfo.get('error'):
            return JsonResponse({'code':12345,'error':'unable get token'})
        weibo_uid = userInfo.get('uid', None)
        access_token = userInfo.get('access_token', None)
        try:
            weibo_user = WeiboProfile.objects.get(wuid=weibo_uid)
        except Exception as e:
            # 如果查不到相关的token 则说明该用户第一次来
            WeiboProfile.objects.create(access_token=access_token, wuid=weibo_uid)
            data = {
                'code': 201,
                'uid': weibo_uid
            }
            return JsonResponse(data)
        else:
            # 如果查询到之前来过，检查是否绑定过
            user = weibo_user.user_profile
            if user:
                #之前成功绑定
                username = user.username
                token = make_token(username)
                return JsonResponse({'code': 200, 'username': username, 'token': token.decode()})
            else:
                #未绑定
                data = {
                    'code': 201,
                    'uid': weibo_uid
                }
                return JsonResponse(data)

    def post(self, request):
        """
        用户提交了关于个人信息以及uid，创建用户，并且创建绑定微博关系
        """
        data = json.loads(request.body)
        uid = data.get('uid', None)
        username = data.get('username', None)
        password = data.get('password', None)
        phone = data.get('phone', None)
        email = data.get('email', None)
        # 创建用户表
        m = hashlib.md5()
        m.update(password.encode())
        # 创建用户以及微博用户表
        try:
            with transaction.atomic():
                user = UserProfile.objects.create(username=username, password=m.hexdigest(),
                                       email=email, phone=phone)
                weibo_user = WeiboProfile.objects.get(wuid=uid)
                weibo_user.user_profile = user
                weibo_user.save()
        except Exception as e:
            return JsonResponse({'code': 10128, 'error':'create user failed!'})
        token = make_token(username)
        return JsonResponse({'code': 200, 'username': username, 'token': token.decode()})


class Users(View):
    def post(self, request):
        """
        Cautions,verify_url :此处发送前端的地址。
        """
        data = json.loads(request.body)
        username = data.get('uname')
        email = data.get('email')
        password = data.get('password')
        phone = data.get('phone')
        carts_data = data.get('carts')
        if len(password) < 6 or len(password) > 12:
            result = {'code': 101990, 'error': 'Password length is wrong'}
            return JsonResponse(result)

        if len(username) < 6 or len(username) > 11:
            result = {'code': 101991, 'error': 'Username length is wrong'}
            return JsonResponse(result)

        if carts_data:
            carts_data = json.loads(carts_data)
        # 优先查询当前用户名是否已存在
        old_user = UserProfile.objects.filter(username=username)
        if old_user:
            result = {'code': 10199, 'error': 'Your username is already existed'}
            return JsonResponse(result)
        m = hashlib.md5()
        m.update(password.encode())
        try:
            user = UserProfile.objects.create(username=username, password=m.hexdigest(),email=email, phone=phone)
        except Exception as e:
            return JsonResponse({'code': 10198, 'error': 'Server is busy'})
        # 此链接是通过用户名和code中间拼接了/
        try:
            code = "%d" % random.randint(1000, 9999)
            code_str = code + '/' + username
            # 生成激活链接：
            active_code = base64.urlsafe_b64encode(code_str.encode(encoding='utf-8')).decode('utf-8')
            # TODO : 用户激活链接永久有效，可以根据自己的喜好去设置。
            EMAIL_CACHE.set("email_active_%s" % username, code, 60*60*24*3)
            verify_url = settings.CLIENT_HOST + '/dadashop/templates/active.html?code=%s' % (active_code)
            token = make_token(username)
            send_active_email.delay(email, verify_url)
            #合并购物车
            carts_obj = CartsView()
            carts_len = carts_obj.merge_carts(user.id,carts_data)
            #res = merge_cart(user,token,cart_data)
        except Exception as e:
            return JsonResponse({'code':10111,'error':'something bad!'})

        return JsonResponse({'code': 200, 'username': username, 'data':{'token': token.decode()},'carts_count':carts_len})



