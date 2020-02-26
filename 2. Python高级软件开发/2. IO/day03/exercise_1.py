"""
使用dict.txt文件完成单词的查找
1.使用input输入一个单词
2.查找到这个单词，打印出该单词及其解释

温馨提示： 每个单词占一行，
        单词与解释之间有空格,
        单词按照从小到大的顺序排列
         借助于split()

思路：  逐行遍历文件
       对每一行用split()提取单词进行比对
       找到单词就打印这一行
       最后找不到给个提示
"""

word = input("Word:")  # 要查找的单词

# 打开文件
f = open("dict.txt")  # 默认就是r方式

# 每次获取一行
for line in f:
    w = line.split(' ')[0]  # 提取每一行的单词
    # 提高一点效率 如果遍历到的单词已经比目标大就没有必要继续向下找了
    if w > word:
        print("没有找到该单词")
        break
    elif w == word:
        print(line)
        break
else:
    print("没有找到该单词")

f.close()














