"""
统计 words.txt 中每个单词出现的数量
"""
from mrjob.job import MRJob

class WordCount(MRJob):
    def mapper(self, _, line):
        # _: 每行行首的偏移量
        # line: 每行内容
        for word in line.split(' '):
            yield word, 1

    # *****shuffle之前*****
    # hello 1
    # hello 1
    # hello 1
    # hadoop 1
    # ******shuffle之后*****
    # hello 1 1 1
    # hadoop 1
    # twink 1 1

    def reducer(self, word, occur):
        # word: shuffle之后去重的单词
        # occur: 每个单词汇总之后的 hello [1 1 1]
        yield word, sum(occur)

if __name__ == '__main__':
    WordCount.run()












