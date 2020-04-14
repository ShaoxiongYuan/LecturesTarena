import hashlib
import json
import time

from django.http import JsonResponse
from django.shortcuts import render
from user.models import UserProfile
from django.conf import settings

#10200-10299
# Create your views here.
def tokens(request):

    if request.method != 'POST':
        result = {'code':10200, 'error':'Please use POST'}
        return JsonResponse(result)
    # 获取数据
    json_str = request.body
    json_obj = json.loads(json_str)
    username = json_obj['username']
    password = json_obj['password']
    #TODO 校验参数

    #校验用户名 密码
    old_users = UserProfile.objects.filter(username=username)
    if not old_users:
        result = {'code':10201, 'error':'The username or password is wrong!'}
        return JsonResponse(result)

    user = old_users[0]
    m = hashlib.md5()
    m.update(password.encode())
    if m.hexdigest() != user.password:
        result = {'code':10202, 'error':'The username or password is wrong!'}
        return JsonResponse(result)
    #签发token
    token = make_token(username)

    result = {'code':200, 'username':username, 'data':{'token':token.decode()}, 'carts_count':0}
    return JsonResponse(result)


def make_token(username, exp=3600*24):

    import jwt
    now = time.time()
    payload = {'username': username, 'exp': now + exp}
    return jwt.encode(payload, settings.JWT_TOKEN_KEY, algorithm='HS256')