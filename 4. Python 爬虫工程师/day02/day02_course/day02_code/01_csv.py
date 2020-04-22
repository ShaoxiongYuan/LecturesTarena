"""
csv模块使用示例
方法：
    1、单行写入：writerow()   ['','','']
    2、多行写入：writerows()  [(),(),()]
"""
import csv

# newline参数仅仅局限于windows中的空行
with open('spider.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['步惊云','绝世好剑'])

with open('spider.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows([('聂风','血饮狂刀'),('星矢','天马流星拳')])












