from flask import Flask, request

app = Flask(__name__)


# 访问127.0.0.1：5000/ 显示这是项目的首页
# 127.0.0.1:5000/index
@app.route('/')
@app.route('/index')
def index_view():
    return "这是项目的首页"


# 休息16：00～16：15
# 访问127.0.0.1:5000/detail  显示这是项目的详情页
@app.route('/detail')
def detail_view():
    return "这是项目的详情页"


# 127.0.0.1:5000/show/QTX     欢迎QTX
# 127.0.0.1:5000/show/Maira     欢迎Maria
# 127.0.0.1:5000/show/guoxiaonao/18
@app.route('/show/<uname>/<int:age>')
def show_view(uname, age):
    # return "欢迎%s" % uname
    return "姓名%s,年龄%s" % (uname, age)


# http://127.0.0.1:5000/server
@app.route('/server', methods=['GET', 'POST'])
def server():
    # 接收前端提交的数据并显示
    # 通过request对象接收get请求
    # from flask import request
    # print(request.args)#类字典
    # print(request.args['uname'])#通过键索引获取值
    # uname = request.args['uname']

    # 接收POST请求提交的数据
    # uname = request.form['name']
    uname = request.form.get('uname', '')
    return '欢迎%s' % uname


if __name__ == "__main__":
    app.run(debug=True)
