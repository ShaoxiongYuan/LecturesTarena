import requests
import base64
import json

ai_list = {'植物': '/v1/plant', '动物': '/v1/animal', '其他': '/v2/advanced_general'}


def baidu(type, path):
    if type in ai_list:
        url = "https://aip.baidubce.com/rest/2.0/image-classify%s?access_token=24.c36ae190ea9865133bbc1bf1e2d921d4.2592000.1577462210.282335-17874022" % (
            ai_list[type])
    else:
        return None

    header = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {}
    with open(path, 'rb') as f:
        image = base64.b64encode(f.read())
        data['image'] = str(image, 'utf-8')
        res = requests.post(url=url, data=data, headers=header).text
        return json.loads(res).get('result', 'Error')


if __name__ == '__main__':
    print(baidu('动物', 'kl.jpg'))
