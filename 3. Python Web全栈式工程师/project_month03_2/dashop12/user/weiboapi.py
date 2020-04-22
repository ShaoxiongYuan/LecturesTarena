 # -*- coding:utf-8 -*-
import requests
import json
from urllib.parse import urlencode
from django.conf import settings


class OAuthWeibo(object):

    def __init__(self):
        # 网站应用客户端id
        self.client_id = settings.WEIBO_CLIENT_ID
        # 网站应用客户端安全密钥
        self.client_secret = settings.WEIBO_CLIENT_SECRET
        # 网站回调url网址
        self.redirect_uri = settings.WEIBO_REDIRECT_URI

    def get_weibo_login_url(self):
        """
        用于获取微博登陆的URL
        :return: 微博登陆的网址
        """
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
        }

        # 拼接url地址
        weibo_url = 'https://api.weibo.com/oauth2/authorize?'
        # url = 'client_id=123050457758183&redirect_uri=http://www.example.com/response&response_type=code'
        url = weibo_url + urlencode(params)
        return url

    def get_access_token(self, code):
        weibo_token_url = 'https://api.weibo.com/oauth2/access_token'
        req_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'authorization_code',
            'redirect_uri': self.redirect_uri,
            'code': code
        }
        try:
            response = requests.post(weibo_token_url, data=req_data)
        except Exception as e:
            raise Exception('无法获取uid')
        if response.status_code == 200:
            return json.loads(response.text)
        raise Exception('获取uid异常')
