from flask import Flask,render_template,request

#tempaltes 默认的模板目录 放页面的
#static 默认的静态资源目录  放img/css/js...
app = Flask(__name__)

# 练习14：11～14：15
#当用户访问http://127.0.0.1:5000/demo时  给用户显示页面demo1.html
@app.route('/demo')
def demo_view():
    return render_template('demo1.html')


@app.route('/server')
def web_server():
    return '接收前端AJAX请求成功'


#http://127.0.0.1:5000/exer
@app.route('/exer')
def show_exer():
    return render_template('exer1.html')

#http://127.0.0.1:5000/exer_server
@app.route('/exer_server')
def exer_server():
    #接收前端get请求提交的数据 uname
    #将uname的结果返回给前端
    #from flask import request
    uname = request.args.get('uname')
    return '和%s有关的数据' % uname


#http://127.0.0.1:5000/exer2
@app.route('/exer2')
def show_exer2():
    return render_template('exer2.html')

@app.route('/exer2_server')
def exer2_server():
    uname = request.args.get('uname')
    user_list = ['laowang','maria','QTX','shibw']
    if uname in user_list:
        return '1000'#表示用户名已存在
    else:
        return '1001'#表示用户名不存在





if __name__ == "__main__":
    app.run(debug=True)