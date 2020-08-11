from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from .models import Note


# Create your views here.

def logging_check(fn):
    def wrap(request, *args, **kwargs):
        #检查Session
        if 'username' not in request.session or 'uid' not in request.session:
            #检查Cookies
            c_username = request.COOKIES.get('username')
            c_uid = request.COOKIES.get('uid')
            if not c_username or not c_uid:
                #302 跳转到 登录
                return HttpResponseRedirect('/user/login')
            else:
                #Cookies[回写]
                request.session['username'] = c_username
                request.session['uid'] = c_uid

        return fn(request, *args,**kwargs)
    return wrap


@logging_check
def add_view(request):

    if request.method == 'GET':
        return render(request, 'note/add_note.html')

    elif request.method == 'POST':
        #处理文章入库
        title = request.POST['title']
        content = request.POST['content']

        #方案1
        #获取登录用户的 id
        uid = request.session['uid']
        #存
        Note.objects.create(title=title, content=content, user_id=uid)

        #方案2
        # user = User.objects.get(id=uid)
        # Note.objects.create(title=title, content=content, user=user)


        return HttpResponse('添加笔记成功')