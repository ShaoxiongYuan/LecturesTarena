"""
    改造exercise02.py(季度 --> 月份)
    如果输入e键退出,否则循环执行。
    调试：体会循环
"""

while True:
    season = input("请输入季度：")
    if season == "春天":
        print("1月2月3月")
    elif season == "夏天":
        print("4月5月6月")
    elif season == "秋天":
        print("7月8月9月")
    elif season == "冬天":
        print("10月11月12月")
    if input("请输入e键退出：") == "e":
        break
