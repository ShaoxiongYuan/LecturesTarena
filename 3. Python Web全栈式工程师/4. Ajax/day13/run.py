import json
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/getData')
def getData():
    return render_template('getData.html')

@app.route('/getData_server')
def getData_server():
    true = True
    cartdata={
        "username":"dongdong",
        "cart":[
            {
                "id":"1",
                "count":"11",
                "name":"蓝色小尺寸",
                "default_image_url":"2_5pdezhv.jpg",
                "price":100,
                "selected":true,
                "sku_sale_attr_name":["安踏A/尺寸：","安踏A/颜色："],
                "sku_sale_attr_val":["15寸","蓝色"]
            },
            {
                "id":"2",
                "count":"9",
                "name":"红色大尺寸",
                "default_image_url":"3_JGA6F97.jpg",
                "price":10000,
                "selected":true,
                "sku_sale_attr_name":["安踏A/尺寸：","安踏A/颜色："],
                "sku_sale_attr_val":["18寸","绿色"]
            },
            {
                "id":"3",
                "count":"10",
                "name":"蓝色小尺寸",
                "default_image_url":"4_z3FdRMq.jpg",
                "price":1000,
                "selected":true,
                "sku_sale_attr_name":["安踏B/尺寸：","安踏B/颜色："],
                "sku_sale_attr_val":["18寸","蓝色"]
            }
        ]
    }
    # flask1.1.1
    # return cartdata
    # data = json.dumps(cartdata)
    # data = json.dumps({"code":201,"data":"获取数据失败"})
    data = json.dumps({"code":200,"data":cartdata})
    return data


#原本的页面
@app.route('/load')
def loda_view():
    return render_template('load.html')
#远程服务器的页面
@app.route('/load_server',methods=['get','post'])
def load_server():
    # uname = request.args.get('uname')
    # age = request.args.get('age')
    uname = request.form.get('uname')
    age = request.form.get('age')
    print(uname,age)
    return render_template('load_server.html')

@app.route('/header')
def header():
    return render_template('header.html')

@app.route('/footer')
def footer():
    return render_template('footer.html')


#jq-ajax-get
@app.route('/get')
def get_view():
    return render_template('jq_ajax_get.html')

@app.route('/get_server')
def get_server():
    uname = request.args.get('uname')
    age = int(request.args.get('age'))

    if age<18:
        data = {"code":201,"msg":"禁止访问"}
    elif age>=18:
        data = {"code":200,"msg":"允许访问"}
    return json.dumps(data)


@app.route('/post',methods=['get','post'])
def post_view():
    if request.method == 'GET':
        return render_template('jq_ajax_post.html')
    elif request.method == 'POST':
        uname = request.form.get('uname')
        age = request.form.get('age')
        return json.dumps({"code":200,"msg":"获取POST数据成功"})


#练习ajax() 休息15:53~16:10
@app.route('/ajax')
def ajax():
    return render_template('jq_ajax.html')

@app.route('/ajax_server',methods=['get','post'])
def ajax_server():
    if request.method == 'GET':
        import time
        time.sleep(3)
        return json.dumps({"code":200,"msg":"OK"})
    elif request.method == 'POST':
        uname = request.json.get('uname')
        return json.dumps({"code":200,"msg":"欢迎%s"%uname})


#跨域
#http://127.0.0.1:5000/cross
@app.route('/cross')
def cross():
    return render_template('cross.html')

# AssertionError: View function mapping is overwriting an existing endpoint function: cross
# 原因 视图函数命名重复了
@app.route('/cross1')
def cross1():
    return render_template('cross1.html')

@app.route('/cross2')
def cross2():
    return render_template('cross2.html')



if __name__ == "__main__":
    app.run(debug=True)