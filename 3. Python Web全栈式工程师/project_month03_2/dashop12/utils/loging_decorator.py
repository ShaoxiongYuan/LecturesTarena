# -*- coding:utf-8 -*-
#        > File Name: loging_decorator.py
#      > Author: GuoXiaoNao
#     > Mail: 250919354@qq.com
#     > Created Time: Fri 17 May 2019 04:39:34 PM CST


from functools import wraps
from django.http import JsonResponse
import json
import jwt

from user.models import UserProfile



TOKEN_KEY = '1234567'

def logging_check(func):
    @wraps(func)
    def wrapper(self, request, *args, **kwagrs):
        token = request.META.get('HTTP_AUTHORIZATION')
        # print(request.META.get('HTTP_REFERER'))
        if not token:
            result = {'code':403, 'error':'Please login'}
            return JsonResponse(result)

        try:
            res = jwt.decode(token, TOKEN_KEY)
        except Exception as e:
            print('jwt decode error is %s'%(e))
            result = {'code':403, 'error':'Please login'}
            return JsonResponse(result)

        except jwt.ExpiredSignatureError:
            #token过期
            result = {'code':403, 'error':'Please login'}
            return JsonResponse(result)

        username = res['username']
        user = UserProfile.objects.get(username=username)
        request.user = user
        return func(self, request, *args, **kwagrs)
    return wrapper



def get_user_by_request(request):
    '''
    通过request 获取 user
    '''

    token = request.META.get('HTTP_AUTHORIZATION')
    if not token:
        return None

    try:
        res = jwt.decode(token, TOKEN_KEY)
    except Exception as e:
        print('jwt decode error is %s'%(e))
        return None

    except jwt.ExpiredSignatureError:
        #token过期
        return None

    username = res['username']
    user = UserProfile.objects.get(username=username)
    if not user:
        return None

    return user








