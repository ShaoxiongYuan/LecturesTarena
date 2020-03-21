from flask import Flask,request
import json
app = Flask(__name__)

#GET http://127.0.0.1:5001/cross_sever net::ERR_CONNECTION_REFUSED
#原因  防火墙/网络问题   或者 服务没启动

# ValueError: urls must start with a leading slash
# 原因 路由地址必须以 / 开始
@app.route('/cross_server')
def cross_server():
    fun = request.args.get('callback')
    print(fun)
    data = {"code":200,"msg":"OK"}
    #"print()"
    print(fun+"("+json.dumps(data)+")")
    # 'print({"code": 200, "msg": "OK"})'
    return fun+"("+json.dumps(data)+")"

#休息+练习17：05～17：20
#在 run.py中定路由和视图函数
#访问http://127.0.0.1:5000/cross时能看到cross.html
#在cross.html中向http://127.0.0.1:5001/cross_server发送get请求 获取结果


if __name__ == "__main__":
    #http://127.0.0.1:5001
    app.run(debug=True,port=5001)