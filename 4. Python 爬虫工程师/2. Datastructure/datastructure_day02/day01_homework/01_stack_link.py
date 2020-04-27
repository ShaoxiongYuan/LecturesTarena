"""
使用链式存储实现栈
思路:
1. 栈特点: 后进先出,所有的操作都再栈顶
2. 封装方法: 入栈 出栈 是否为空 ...
3. 实现:  把单链表的头部作为栈顶（入栈、出栈）
"""
class Node:
    """节点类"""
    def __init__(self,value):
        self.value = value
        self.next = None

# 链式存储栈结构
class LinkStack:
    def __init__(self):
        # 创建空栈,链表头部作为栈顶,top可理解为链表的头节点
        self.top = None

    def is_empty(self):
        """判断是否为空栈"""
        if self.top is None:
            return True
        return False

    def push(self,val):
        """入栈: 相当于在链表头部添加节点"""
        node = Node(val)
        node.next = self.top
        self.top = node

    def pop(self):
        """出栈: 相当于删除头节点"""
        if self.is_empty():
            raise Exception('pop from empty stack')

        val = self.top.value
        self.top = self.top.next

        return val

    def stack_top(self):
        """查看栈顶元素: 查看头节点"""
        if self.is_empty():
            return None
        return self.top.value

    def size(self):
        """查看栈的大小"""
        if self.is_empty():
            return 0
        count = 0
        current = self.top
        while current is not None:
            current = current.next
            count += 1

        return count

if __name__ == '__main__':
    s = LinkStack()
    # 栈顶->栈底: 300 200 100
    s.push(100)
    s.push(200)
    s.push(300)
    # 出栈: 300
    print(s.pop())
    # 查看栈顶元素: 200
    print(s.stack_top())
    # 栈大小: 2
    print(s.size())





