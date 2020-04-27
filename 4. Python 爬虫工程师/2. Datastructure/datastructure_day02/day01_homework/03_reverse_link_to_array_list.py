"""
输入一个链表，按链表值从尾到头的顺序返回一个 array_list
解题思路: 将链表中从头节点开始依次取出节点元素，append到array_list中，并进行最终反转
"""

class Node:
    """节点类"""
    def __init__(self,value):
        self.value = value
        self.next = None

# 解决方案
class Solution:
    # 返回从链表尾部到头部的序列, node为头结点
    def get_array_list(self,node):
        array_list = []
        while node is not None:
            array_list.append(node.value)
            node = node.next

        # 将最终列表进行反转,reverse()无返回值,直接改变列表
        array_list.reverse()

        return array_list

if __name__ == '__main__':
    s = Solution()
    # 链表(表头->表尾): 100 200 300
    n1 = Node(100)
    n1.next = Node(200)
    n1.next.next = Node(300)
    # 调用反转方法: [300, 200, 100]
    array_list = s.get_array_list(n1)
    print(array_list)







