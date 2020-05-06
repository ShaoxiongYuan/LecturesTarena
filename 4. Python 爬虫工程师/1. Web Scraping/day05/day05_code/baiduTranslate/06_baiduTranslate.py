"""
百度翻译结果数据抓取
"""
import requests
import execjs

class BaiduTranslateSpider:
    def __init__(self):
        self.url = 'https://fanyi.baidu.com/v2transapi?from=en&to=zh'
        self.headers = {
            'Cookie':'BIDUPSID=46D0471B72D849FC7EDF21BA4702F83C; PSTM=1587698693; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BAIDUID=46D0471B72D849FCE9A270A451DF87D1:SL=0:NR=10:FG=1; H_PS_PSSID=30969_1463_31326_21107_31427_31341_31228_30824_31164; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1587727393,1587785676; delPer=0; PSINO=2; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1587795459; __yjsv5_shitong=1.0_7_6a5f66b7527ef7c72b25325159665a94b252_300_1587795459522_101.30.19.86_e355ed00; yjs_js_security_passport=83337e9fc826bc1032882150592c0ad37f63fbb4_1587795460_js',
            'Referer':'https://fanyi.baidu.com/',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36',
        }

    def get_sign(self, word):
        """获取sign - form表单中只缺少这一个数据"""
        with open('translate.js', 'r', encoding='utf-8') as f:
            js_code = f.read()

        loader = execjs.compile(js_code)
        return loader.call('e', word)

    def translate(self, word):
        """具体翻译函数"""
        sign = self.get_sign(word)
        data = {
            "from": "en",
            "to": "zh",
            "query": word,
            "transtype": "translang",
            "simple_means_flag": "3",
            "sign": sign,
            "token": "e18b978124aedcb071caa317732a5fc4",
            "domain": "common",
        }
        html = requests.post(url=self.url, data=data, headers=self.headers).json()

        return html['trans_result']['data'][0]['dst']

    def run(self):
        word = input('请输入要翻译的单词:')
        print(self.translate(word))

if __name__ == '__main__':
    spider = BaiduTranslateSpider()
    spider.run()
















