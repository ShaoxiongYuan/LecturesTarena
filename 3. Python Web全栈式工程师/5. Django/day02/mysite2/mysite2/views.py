from django.http import HttpResponse
from django.shortcuts import render

def test_html(request):

    s = request.POST.get('sss')
    # from django.template import loader
    # #1 加载html
    # t = loader.get_template('test_html.html')
    # #2 执行render 转化成字符串
    # html = t.render()
    # #3 响应给浏览器
    # return HttpResponse(html)
    dic = {
        'username': 'guoxiaonao',
        'age':18,
        'lst': [],
        'd': {'title':'xiaonao'},
        'func': say_hi,
        'class_obj': Dog(),
        'script': '<script>alert(11)</script>'
    }

    #xss(Cross Site Script)注入 用户通过网站输入框，输入一段JS代码;网站接收到js代码之后;执行了代码中的相应步骤，从而遭到的损失;
    #防范 -  转义用户输入 import html
    # html.escape('<script>alert(111)</script>')
    return render(request,'test_html.html', dic)

class Dog:
    def say(self):
        return 'hahahaha'

def say_hi():

    return 'Hi everyone'

def mycal_view(request):

    if request.method == 'GET':
        return render(request, 'mycal.html')

    elif request.method == 'POST':
        #计算数据
        x = request.POST['x']
        y = request.POST['y']
        op = request.POST['op']

        try:
            x = int(x)
            y = int(y)
        except Exception as e:
            error = 'The input is error'
            return render(request, 'mycal.html', locals())
            #return HttpResponse('The input is error')

        if op == 'add':
            result = x + y
        elif op == 'sub':
            result = x - y
        elif op == 'mul':
            result = x * y
        elif op == 'div':
            if y == 0:
                return HttpResponse('The input is error~')
            result = x / y
        else:
            return HttpResponse('The op is error~')
        #dic = {}
        #dic['op'] = op

        return render(request, 'mycal.html', locals())

























