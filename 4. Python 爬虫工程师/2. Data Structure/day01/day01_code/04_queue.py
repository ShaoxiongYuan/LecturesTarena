"""
使用链式存储实现队列 - 先进先出(FIFO)
思路:
    1、队尾进行入队操作,队头进行出队操作
    2、链表尾部作为队尾,进行入队操作
       链表头部作为队头,进行出队操作
    3、入队操作: 在链表尾部添加1个节点
       出队操作: 在链表头部删除1个节点
"""
class Node:
    """节点类"""
    def __init__(self, value):
        self.value = value
        self.next = None

class LQueue:
    def __init__(self):
        # 初始化1个空队列
        self.head = None

    def is_empty(self):
        """判断队列是否为空"""
        return self.head == None

    def enqueue(self, item):
        """队尾入队 - 相当于在链表尾部添加一个节点"""
        node = Node(item)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            # 循环结束后,cur指向尾节点
            cur.next = node
            node.next = None

    def dequeue(self):
        """队头出队 - 相当于删除链表头节点"""
        if self.is_empty():
            raise Exception('dequeue from empty lqueue')
        # 删除链表头节点
        item = self.head.value
        self.head = self.head.next

        return item

if __name__ == '__main__':
    q = LQueue()
    # 队列: 队头->队尾  100 200 300
    q.enqueue(100)
    q.enqueue(200)
    q.enqueue(300)
    # 终端1: False
    print(q.is_empty())
    # 终端2: 100
    print(q.dequeue())
    # 终端3: 200
    print(q.dequeue())
    # 终端4: 300
    print(q.dequeue())
    # 终端5: dequeue from empty lqueue
    print(q.dequeue())















