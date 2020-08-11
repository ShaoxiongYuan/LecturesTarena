from django.http import HttpResponse, HttpResponseRedirect



POST_FORM = '''
<form method='post' action="/test_get_post">
    姓名:<input type="text" name="username">
    <input type='submit' value='登陆'>
</form>
'''



def page_2003_view(request):


    return HttpResponse('这是page2003页面')


def index_view(request):

    return HttpResponse('这是首页')

def page1_view(request):

    return HttpResponse('这是编号1的网页')

def page2_view(request):

    return HttpResponse('这是编号2的网页')


def pagen_view(request, pg):

    html = '这是编号%s的网页!!!!'%(pg)
    return HttpResponse(html)


def cal_view(request, num1, op, num2):

    if op not in ['add', 'sub','mul']:
        return HttpResponse('Your op is wrong~')

    result = 0
    if op == 'add':
        result = num1 + num2
    elif op == 'sub':
        result = num1 - num2
    elif op == 'mul':
        result = num1 * num2

    return HttpResponse('结果为%s'%(result))


def birthday_view(request, y, m, d):

    print('path_info is', request.path_info)
    print('method is', request.method)
    print('request GET is', request.GET)
    print('full path is', request.get_full_path())



    html = '生日为:%s年%s月%s日'%(y, m , d)
    #return HttpResponse(html)

    #302跳转 传入 '/'开头的相对地址 http://127.0.0.1:8000/page/1
    # 301/302 响应 会包含 Location 响应头, 值为 要跳转的地址,浏览器接到 301/302 时,检查 Location 头,把用户地址栏改写成对应的地址,发出第二个请求(GET)
    return HttpResponseRedirect('/page/1')


def test_get_post(request):

    #?a=100&b=20&c=300
    #print('query string is', request.GET['c'])
    #print(request.GET.get('z', 'no key!!'))
    #?a=100&b=200&a=300
    # print(request.GET)
    # print(request.GET['a'])
    # print(request.GET.getlist('a'))
    # return HttpResponse('test get post is ok')
    if request.method == 'GET':
        return HttpResponse(POST_FORM)
    elif request.method == 'POST':
        #处理数据
        #表单post提交 请求头Content-Type:application/x-www-form-urlencoded
        #request.POST 只能取表单提交的post数据
        value = request.POST['username']
        return HttpResponse('--%s post is ok--'%(value))









