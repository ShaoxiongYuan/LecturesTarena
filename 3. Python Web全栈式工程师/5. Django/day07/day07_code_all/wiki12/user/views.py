from django.http import HttpResponse
from django.shortcuts import render
from .models import User
import hashlib

# Create your views here.
def reg_view(request):

    if request.method == 'GET':

        return render(request, 'user/register.html')

    elif request.method == 'POST':

        username = request.POST.get('username')
        password_1 = request.POST.get('password_1')
        password_2 = request.POST.get('password_2')

        #TODO 参数的检查

        if password_1 != password_2:
            return HttpResponse('The password is not same')

        #检查一下用户名是否存在
        old_users = User.objects.filter(username=username)
        if old_users:
            return HttpResponse('The username is already existed')

        #将明文密码转化成哈希值
        #哈希算法
        #1,定长定值的输出 -> 算法不变的情况下，无论输入是多少，输出定长/ 算法不变的情况下，输入不变，则输出的值也不变
        #2,不可逆 -> hash结果,反算不出明文结果
        #3,雪崩效应 -> 明文改变，则输出值一定改变 - 文件完整性校验[大文件抽样]


        m = hashlib.md5()
        m.update(password_1.encode())
        passwrd_h = m.hexdigest()

        try:
            #由于username有唯一索引，所以此处可能会有插入异常
            user = User.objects.create(username=username,password=passwrd_h)
        except Exception as e:
            print('---create error', e)
            return HttpResponse('---create error----')

        #保存状态
        request.session['uid'] = user.id
        request.session['username'] = username

        return HttpResponse('---register is ok---')


def login_view(request):

    if request.method == 'GET':
        #优先检查session
        if 'username' in request.session and 'uid' in request.session:
            return HttpResponse('---您已登录 !---')
        #检查Cookies
        username = request.COOKIES.get('username')
        uid = request.COOKIES.get('uid')
        if username and uid:
            #回写session
            request.session['username'] = username
            request.session['uid'] = uid
            return HttpResponse('---您已登录 !!---')

        return render(request, 'user/login.html')

    elif request.method == 'POST':

        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            old_user = User.objects.get(username=username)
        except Exception as e:
            return HttpResponse('The username or password is wrong~')

        m = hashlib.md5()
        m.update(password.encode())
        password_h = m.hexdigest()

        #比对hash值
        if password_h != old_user.password:
            return HttpResponse('The username or password is wrong~')

        #存储状态
        request.session['uid'] = old_user.id
        request.session['username'] = username
        print('----login-post----')
        print(request.POST)

        #判断保持Cookies
        resp = HttpResponse('---login is ok---')
        if 'remember' in request.POST:
            resp.set_cookie('username',username, 3600*24*3)
            resp.set_cookie('uid', old_user.id, 3600*24*3)

        return resp



















































