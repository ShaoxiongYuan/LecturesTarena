from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader


def test_html(request):
    #方案1 loader
    #1,加载html
    #t = loader.get_template('test_html.html')
    #2,生成html的字符串
    #html = t.render()
    #3,return
    #return HttpResponse(html)

    #方案2

    dic = {}
    dic['str'] = 'guoxiaonao'
    dic['int'] = 18
    dic['lst'] = ['Jack', 'Tom', 'Lily']
    #dic['lst'] = []
    dic['d'] = {'name':'guoxiaonao', 'desc':'haha'}
    dic['function'] = say_hi
    dic['class_obj'] = Dog()
    dic['script'] = '<script>alert(11)</script>'

    return render(request, 'test_html.html', dic)



def say_hi():
    return 'hahahahaha'

class Dog():

    def say(self):
        return 'wangwang'


def mycal_view(request):

    if request.method == 'GET':
        #GET返回页面
        return render(request, 'mycal.html')
    elif request.method == 'POST':
        #处理数据
        #POST表单提交数据的结构 跟 GET查询字符串的结构一致 [key=value&key2=value2] ; POST将数据放在请求的body里;
        x = request.POST['x']
        y = request.POST['y']
        if not x or not y:
            return HttpResponse('Please give me number')

        try:
            x = int(x)
            y = int(y)
        except Exception as e:
            print('error is %s'%(e))
            return HttpResponse('Please give me number !!')

        op = request.POST['op']

        result = 0
        if op == 'add':
            result = x + y
        elif op == 'sub':
            result = x - y
        elif op == 'mul':
            result = x * y
        elif op == 'div':
            #TODO 检查参数
            result = x / y
        #locals() 返回含有当前局部变量的字典
        return render(request, 'mycal.html', locals())


def base_view(request):
    lst = ['Jay']
    return render(request, 'base.html', locals())

def music_view(request):

    return render(request, 'music.html')

def sport_view(request):

    from django.urls import reverse
    print('url is', reverse('music_url'))
    return HttpResponseRedirect(reverse('music_url'))
    #return render(request, 'sport.html')









