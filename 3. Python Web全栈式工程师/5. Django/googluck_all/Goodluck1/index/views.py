from django.shortcuts import render
import random
# Create your views here.
prize_list = ['/static/images/prize1.jpg',
        '/static/images/prize2.jpg',
        '/static/images/prize3.jpg',
        '/static/images/prize4.jpg',
        '/static/images/prize5.jpg',
        '/static/images/prize6.jpg',
        '/static/images/prize7.jpg',
        '/static/images/prize8.jpg']
def get_page(request):
    # 测试,模板与静态文件是否可用
    if request.method == 'GET':
        # 随机奖品顺序
        random.shuffle(prize_list)
        # 把九张图片全部通过模板标签显示到index页
        return render(request, 'index.html',{'prizes':prize_list})
# 获取抽奖结果
def get_prize(request):
    random_num = random.randint(0,len(prize_list)-1)
    prize = prize_list[random_num]
    return render(request,'prize.html',{'prize':prize})