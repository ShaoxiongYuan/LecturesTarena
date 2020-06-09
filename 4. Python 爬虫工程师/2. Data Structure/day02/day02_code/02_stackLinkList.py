"""
使用链式存储结构实现栈模型 - LIFO
思路：
    1、设计链表的头部作为栈顶,进行入栈和出栈操作
    2、链表的尾部作为栈底,不进行任何操作
    3、入栈: 在链表的头部添加1个节点
       出栈: 在链表的头部删除1个节点
"""
class Node:
    """节点类"""
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkStack:
    def __init__(self):
        """初始化一个空栈"""
        self.head = None

    def is_empty(self):
        """判断是否为空栈"""
        return self.head == None

    def push(self, item):
        """入栈操作: 相当于在链表头部添加1个节点"""
        node = Node(item)
        node.next = self.head
        self.head = node

    def pop(self):
        """出栈操作: 相当于在链表头部删除1个节点"""
        if self.is_empty():
            raise Exception('pop from empty stack')
        item = self.head.value
        self.head = self.head.next

        return item

    def size(self):
        """查看栈的大小"""
        count = 0
        cur = self.head
        while cur:
            cur = cur.next
            count += 1

        return count

if __name__ == '__main__':
    s = LinkStack()
    # 栈: 300 200 100
    s.push(100)
    s.push(200)
    s.push(300)
    # 终端1: 300
    print(s.pop())
    # 终端2: 2
    print(s.size())
    # 终端3: False
    print(s.is_empty())









