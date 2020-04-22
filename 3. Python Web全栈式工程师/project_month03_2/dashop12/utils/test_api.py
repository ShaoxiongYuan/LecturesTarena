import threading
import requests
import random

#python3 manger.py runserver 8001

def getRequest():

    url='http://127.0.0.1:8000/v1/users/test_api'
    url2='http://127.0.0.1:8001/v1/users/test_api'
    get_url = random.choice([url, url2])
    requests.get(get_url)

ts = []
for i in range(30):

    t=threading.Thread(target=getRequest,args=())
    ts.append(t)

if __name__ == '__main__':

    for t in ts:
        t.start()

    for t in ts:
        t.join()
