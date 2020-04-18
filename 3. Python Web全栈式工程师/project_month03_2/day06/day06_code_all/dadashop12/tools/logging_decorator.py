import jwt
from django.http import JsonResponse
from django.conf import settings
from user.models import UserProfile

def logging_check(func):
    def wrapper(self, request, *args, **kwargs):
        #获取token
        token = request.META.get('HTTP_AUTHORIZATION')
        if not token:
            return JsonResponse({'code':403, 'error':'Please login'})
        #decode token
        try:
            res = jwt.decode(token, settings.JWT_TOKEN_KEY, algorithms='HS256')
        except Exception as e:
            print('jwt decode error is %s'%(e))
            return JsonResponse({'code':403, 'error':'Please login'})

        # payload username -> 获取用户 -> 赋值给request.myuser
        username = res['username']
        user = UserProfile.objects.get(username=username)
        request.myuser = user

        return func(self, request, *args, **kwargs)
    return wrapper

