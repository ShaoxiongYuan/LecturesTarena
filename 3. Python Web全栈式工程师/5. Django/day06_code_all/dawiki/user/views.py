from django.http import HttpResponse
from django.shortcuts import render
from .models import User
import hashlib


# Create your views here.
def reg_view(request):

    if request.method == 'GET':
        return render(request, 'user/register.html')
    elif request.method == 'POST':
        #处理注册
        username = request.POST['username']
        password_1 = request.POST['password_1']
        password_2 = request.POST['password_2']

        if not username or not password_1:
            return HttpResponse('Please give me data ~')

        if password_1 != password_2:
            return HttpResponse('The password is error ~')

        #判断用户名是否可用
        old_user = User.objects.filter(username=username)
        if old_user:
            return HttpResponse('The username is already existed')

        #密码处理
        #hash算法
        #1,算法恒定 定长输出;
        #2,不可逆 - 无法通过hash值反算回明文
        #3,雪崩效益 - 输入变化 输出一定变化
        #场景 - 1,密码在数据库中的存储
        #      2,文件完整性校验 - 经典面试题 - 如何做大文件校验 40G 8G - 抽样
        m = hashlib.md5()
        m.update(password_1.encode())
        password_h = m.hexdigest()

        #插入数据 - 并发写入相同用户名, 第一个到达mysql的写入请求成功写入;其余报错 - '重复插入'
        try:
            user = User.objects.create(username=username, password=password_h)
        except Exception as e:
            print('create error is %s'%(e))
            return HttpResponse('The username is already existed !!')

        #TODO 免登陆一天

        return HttpResponse('注册成功')


def login_view(request):

    if request.method == 'GET':
        #检查登录状态 , 如果登录显示 您已登录, 否则显示登录html
        #优先检查session
        if 'username' in request.session and 'uid' in request.session:
            return HttpResponse('--您已登录')
        #检查Cookies
        username = request.COOKIES.get('username')
        uid = request.COOKIES.get('uid')
        if username and uid:
            #回写session
            request.session['username'] = username
            request.session['uid'] = uid
            return HttpResponse('--您已登录')

        return render(request, 'user/login.html')

    elif request.method == 'POST':
        #处理数据
        username = request.POST['username']
        password = request.POST['password']

        try:
            old_user = User.objects.get(username=username)
        except Exception as e:
            print('login get error is %s'%(e))
            return HttpResponse('The username or password is wrong~')

        m = hashlib.md5()
        m.update(password.encode())
        password_h = m.hexdigest()

        if password_h != old_user.password:
            return HttpResponse('The username or password is wrong~!')

        #存储会话状态
        #存session
        request.session['uid'] = old_user.id
        request.session['username'] = old_user.username

        #判断是否需要存储Cookies;
        resp = HttpResponse('登陆成功')
        if 'remember' in request.POST:
            resp.set_cookie('uid', old_user.id, 3600*24*3)
            resp.set_cookie('username', username, 3600*24*3)

        return resp


def logout_view(request):

    #删session
    if 'username' in request.session:
        del request.session['username']
    if 'uid' in request.session:
        del request.session['uid']

    #删Cookies
    resp = HttpResponse('--已登出')
    if 'username' in request.COOKIES:
        resp.delete_cookie('username')
    if 'uid' in request.COOKIES:
        resp.delete_cookie('uid')

    return resp



















