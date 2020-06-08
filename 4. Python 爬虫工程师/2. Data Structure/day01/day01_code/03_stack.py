"""
使用顺序存储的方式来实现栈模型
思路：
    1、栈特点：后进先出,只能在一端进行操作
    2、思考：
       2.1) 可以使用顺序存储方式来实现
       2.2) 可以使用链式存储方式来实现
    3、决定先使用顺序存储实现（栈中元素在内存中是连续的）
       列表尾部作为栈顶,进行入栈（append()）和出栈(pop())操作
       列表头部作为栈底,不进行任何操作
"""
class Stack:
    def __init__(self):
        # 初始化一个空栈
        self.elems = []

    def is_empty(self):
        """判断是否为空栈"""
        return self.elems == []

    def push(self, value):
        """入栈操作 - 相当于在列表的尾部添加一个元素"""
        self.elems.append(value)

    def destack(self):
        """出栈操作 - 相当于在列表的尾部弹出一个元素"""
        if self.is_empty():
            raise Exception('pop from empty stack')
        return self.elems.pop()

    def travel(self):
        """查看栈中所有元素"""
        for elem in self.elems:
            print(elem, end=" ")
        print()

    def top(self):
        """查看栈顶元素 - 相当于查看列表中的最后1个元素"""
        if self.is_empty():
            raise Exception('stack is empty')
        return self.elems[-1]

if __name__ == '__main__':
    s = Stack()
    # 栈: 栈底->栈顶: 100 200 300
    s.push(100)
    s.push(200)
    s.push(300)
    # 终端1: 300
    print(s.top())
    # 终端2: 100 200 300
    s.travel()
    # 终端3: 300
    print(s.destack())
    # 终端4: False
    print(s.is_empty())


















