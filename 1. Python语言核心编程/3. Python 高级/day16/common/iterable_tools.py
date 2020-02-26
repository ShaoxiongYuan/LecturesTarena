"""
    可迭代对象的工具
"""


class IterableHelper:
    """
        可迭代对象助手类
    """

    @staticmethod
    def find_all(iterable, func_condition):
        """
            在可迭代对象中搜索满足条件的元素
        :param iterable:需要搜索的可迭代对象
        :param func_condition:函数类型,搜索条件
        :return:生成器,可以推算出满足条件的数据
        """
        for item in iterable:
            if func_condition(item):
                yield item

    @staticmethod
    def find(iterable, func_condition):
        """

        :param iterable:
        :param func_condition:
        :return:
        """
        for item in iterable:
            if func_condition(item):
                return item

    @staticmethod
    def select(iterable, func_handle):
        for item in iterable:
            # yield item.name
            # yield item.name,item.face_score
            yield func_handle(item)

    @staticmethod
    def get_max(iterable, func_condition):
        max_value = iterable[0]
        for i in range(1, len(iterable)):
            # if max_value.money < iterable[i].money:
            if func_condition(max_value) < func_condition(iterable[i]):
                max_value = iterable[i]
        return max_value

    @staticmethod
    def order_by(iterable, func_condition):
        for r in range(len(iterable) - 1):
            for c in range(r + 1, len(iterable)):
                # if iterable[r].face_score > iterable[c].face_score:
                if func_condition(iterable[r]) > func_condition(iterable[c]):
                    iterable[r], iterable[c] = iterable[c], iterable[r]
