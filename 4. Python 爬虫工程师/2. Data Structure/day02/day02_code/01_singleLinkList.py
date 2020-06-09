"""
Python实现单链表
思路：
    1、链表中每个节点会有2个属性,1个为数据,1个为指针,需要一个节点类
    2、创建单链表的类: 数学模型
       定义方法:      在数学模型的基础上定义一组操作
    3、一组操作:
       3.1) 判断链表是否为空链表
       3.2) 获取链表长度
       3.3) 遍历整个链表
       3.4) 在链表头部添加1个节点
       3.5) 在链表尾部添加1个节点
"""
class Node:
    """节点类"""
    def __init__(self, value):
        self.value = value
        self.next = None

class SingleLinkList:
    """单链表类"""
    def __init__(self):
        self.head = None

    def is_empty(self):
        """判断链表是否为空"""
        return self.head == None

    def length(self):
        """获取链表长度"""
        # 从头节点开始,依次遍历,一直到尾节点
        cur = self.head
        count = 0
        while cur:
            count += 1
            cur = cur.next

        return count

    def add(self, value):
        """链表头部添加1个节点"""
        # 1、把新加的节点的指针指向原来头节点
        # 2、再把新加的节点设置为新的头节点
        node = Node(value)
        node.next = self.head
        self.head = node

    def travel(self):
        """遍历整个链表"""
        cur = self.head
        while cur:
            print(cur.value, end=" ")
            cur = cur.next
        print()

    def append(self, value):
        """在链表尾部添加1个节点"""
        node = Node(value)
        # 特殊情况：空链表
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            # 循环结束后,cur指向了尾节点
            cur.next = node
            node.next = None

    def search(self, value):
        """查看链表中是否存在value数据"""
        cur = self.head
        while cur:
            if cur.value == value:
                return True
            else:
                cur = cur.next
        # 整个链表中不存在value
        return False

    def insert(self, postion, value):
        """在指定位置插入1个节点,postion从0开始"""
        if postion < 0:
            self.add(value)
        elif postion > self.length()-1:
            self.append(value)
        else:
            pre = self.head
            count = 0
            while count < postion-1:
                pre = pre.next
                count += 1
            # 循环结束后,pre指向position前一个节点
            node = Node(value)
            node.next = pre.next
            pre.next = node

if __name__ == '__main__':
    s = SingleLinkList()
    # 链表: 100 -> 200 -> 300
    s.add(300)
    s.add(200)
    s.add(100)
    # 终端1: False
    print(s.is_empty())
    # 终端2: 3
    print(s.length())
    # 终端3: 100 200 300
    s.travel()
    # 链表: 100 -> 200 -> 300 -> 400 -> None
    s.append(400)
    # 终端4: 100 200 300 400
    s.travel()
    # 终端5: True
    print(s.search(300))
    # 终端6: False
    print(s.search(666))
    s.insert(2, 600)
    # 终端7: 100 200 600 300 400
    s.travel()




