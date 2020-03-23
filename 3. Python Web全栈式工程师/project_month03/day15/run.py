from flask import Flask,render_template,request
import json
import time

app = Flask(__name__,template_folder='templates',static_folder='static')

@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/data')
def get_data():
    size = request.args.get('size')#字符串
    with open('comment.data','r') as f:
        all_data = json.loads(f.read())
        #如果是第一次请求
        if not size:
            datas = all_data[:10]
        else:
            size = int(size)
            datas = all_data[size:size+10]
        
        if datas:
            return json.dumps({"code":200,"data":datas})
        else:
            return json.dumps({"code":201,"error":"没有数据了,别刷了,求你了!"})



@app.route('/add',methods=['post'])
def add_data():
    username = request.json.get('username')
    content = request.json.get('content')
    with open('comment.data','r') as f:
        all_data = json.loads(f.read())
    with open('comment.data','w') as f:
        new_data = {
            'num':len(all_data)+1,
            'username':username,
            'time':time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()),
            'content':content
        }
        all_data.append(new_data)
        json_str = json.dumps(all_data)
        f.write(json_str)
    return json.dumps({"code":200,"msg":'内容发布成功'})




if __name__ == "__main__":
    app.run(debug=True)