from flask import Flask,request,render_template


#Flask(import_name,template_folder="templates")
#模板就是包含了特殊语法的HTML
#Flask中所有的模板默认放在templates目录中 如果templates目录没有指定，需要手动创建
#自定义模板目录  在文件中修改模板目录后 记得在目录结构中创建相对应的目录
app = Flask(__name__,template_folder="temp")
#10:50~11:05
#127.0.0.1:5000/
@app.route('/')
def index_view():
    return render_template('review.html')

#9：50～10：05
#127.0.0.1:5000/server
@app.route('/server',methods=['get','post'])
def server():
    #request.method 保存当前请求的数据提交方式  "GET"/"POST"
    #request.args  接收get方式提交的数据
    #request.form  接收post方式提交的数据
    if request.method == 'GET':
        uname = request.args.get('uname')
        return '欢迎%s' % uname
    elif request.method == 'POST':
        uname = request.form.get('uname')
        return '欢迎%s' % uname

#http://127.0.0.1:5000/test/all  使用路由传参
#http://127.0.0.1:5000/test?level=all   使用get方式传参
@app.route('/test')
def test_view():
    level = request.args.get('level')
    if level == "all":
        return "全部难度的内容"
    elif level == "easy":
        return "初级难度的内容"
    elif level == "mid":
        return "中级难度的内容"
    elif level == "hard":
        return "高级难度的内容"
    else:
        return "获取数据失败"


if __name__ == "__main__":
    app.run(debug=True)#在开发阶段打开DEBUG  上线关闭