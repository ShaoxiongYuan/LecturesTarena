import json
import base64
import time
import hmac
import copy

class Jwt():

    def __init__(self):
        pass


    @staticmethod
    def encode(my_payload, key, exp=300):

        #init header
        header = {'typ': 'JWT', 'alg': 'HS256'}
        header_json = json.dumps(header, separators=(',',':'), sort_keys=True)
        #separators: 第一个参数为 每个键值对之间拿什么分割， 第二个参数为每个键与值之间拿什么分割
        header_bs = Jwt.b64encode(header_json.encode())

        #init payload
        payload = copy.deepcopy(my_payload)
        payload['exp'] = time.time() + exp
        payload_json = json.dumps(payload, separators=(',',':'),sort_keys=True)
        payload_bs = Jwt.b64encode(payload_json.encode())

        # init sign
        hm = hmac.new(key.encode(), header_bs + b'.' + payload_bs, digestmod='SHA256')
        hm_bs = Jwt.b64encode(hm.digest())

        return header_bs + b'.' + payload_bs + b'.' + hm_bs

    @staticmethod
    def b64encode(j_s):

        return base64.urlsafe_b64encode(j_s).replace(b'=', b'')

    @staticmethod
    def b64decode(b_s):

        rem = len(b_s) % 4
        if rem > 0:
            b_s += b'=' * (4-rem)
        return base64.urlsafe_b64decode(b_s)


    @staticmethod
    def decode(token, key):

        header_bs, payload_bs, sign_bs = token.split(b'.')

        hm = hmac.new(key.encode(), header_bs + b'.' + payload_bs, digestmod='SHA256')
        if Jwt.b64encode(hm.digest()) != sign_bs:
            raise

        #校验时间
        payload_j = Jwt.b64decode(payload_bs)
        payload = json.loads(payload_j)
        exp = payload['exp']
        now = time.time()
        if now > exp:
            raise

        return payload

if __name__ == '__main__':

    import time
    s = Jwt.encode({'username':'guoxiaonao'}, 'abcdef', 3)
    print(s)

    time.sleep(4)

    res = Jwt.decode(s, 'abcdef')
    print(res)























