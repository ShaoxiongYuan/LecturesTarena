"""
用链式存储结构实现队列
思路:
1. 队列特点: 先进先出,队尾入队操作,队头出队操作
2. 使用单链表实现: 尾部添加节点（入队）,头部删除节点（出队）操作
"""
class Node:
    """节点类"""
    def __init__(self,value):
        self.value = value
        self.next = None

class Queue:
    def __init__(self):
        """空队列,记录队头位置"""
        self.head = None

    def is_empty(self):
        """判断队列是否为空"""
        if self.head is None:
            return True
        return False

    def enqueue(self,val):
        """入队:在链表的尾部添加1个节点"""
        node = Node(val)
        if self.is_empty():
            self.head = node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            # 循环结束,current一定是尾节点
            current.next = node
            node.next = None

    def dequeue(self):
        """出队: 删除头节点"""
        if self.is_empty():
            raise Exception('queue is empty')
        # 非空: 删除头节点
        result = self.head.value
        self.head = self.head.next

        return result

    def top(self):
        """查看队头元素: 查看self.head的值"""
        if self.is_empty():
            raise Exception('queue is empty')

        return self.head.value

    def travel(self):
        """遍历整个队列: 从 队头->队尾 输出"""
        if self.is_empty():
            return
        else:
            current = self.head
            while current is not None:
                print(current.value,end=" ")
                current = current.next

            print()

if __name__ == '__main__':
    q = Queue()
    # q:  100 200 300
    q.enqueue(100)
    q.enqueue(200)
    q.enqueue(300)
    # 出队列: 100
    print(q.dequeue())
    # 是否为空: False
    print(q.is_empty())
    # 队头元素: 200
    print(q.top())
    # 遍历: 200 300
    q.travel()
























