from django.http import HttpResponse, JsonResponse,HttpResponseRedirect


def page1_view(request):

    html = '<h1>这是编号为1的网页</h1>'
    #return JsonResponse(html,safe=False)
    #return HttpResponse(html)
    #302跳转依靠响应头中的Location 告知 浏览器 此次跳转的目的地
    return HttpResponseRedirect('/page2')

def page2_view(request):

    html = '<h1>这是编号为2的网页</h1>'
    return HttpResponse(html)

def index_view(request):

    return HttpResponse('这是首页')

def pagen_view(request, n):
    #n -> 为 str
    html = '===这是第%s个页面==='%(n)
    return HttpResponse(html)

def cal_view(request, x, op, y):
    x = int(x)
    y = int(y)
    if op not in ['mul', 'add', 'sub']:
        return HttpResponse('Sorry~Your op is wrong~')

    if op == 'add':
        res = x + y
    elif op == 'mul':
        res = x * y
    else:
        res = x - y

    return HttpResponse('result is %s'%(res))


def person_view(request,name, age):

    return HttpResponse('姓名:%s 年龄%s'%(name, age))


def birthday_view(request, y, m, d):

    print(request.path_info)
    print(request.get_full_path())
    print(request.method)

    #获取包含查询字符串的name和值的 ‘字典’
    print(request.GET)
    print(request.POST)

    return HttpResponse('生日:%s年%s月%s日'%(y,m,d))





