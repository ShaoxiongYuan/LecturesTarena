"""
输入一个链表，按链表值从尾到头的顺序返回一个 array_list
思路:
    1、从头到尾遍历链表,把数据添加到列表中
    2、进行列表反转
"""
class Node:
    """节点类"""
    def __init__(self, value):
        self.value = value
        self.next = None

class Solution:
    def get_array_list(self, head):
        array_list = []
        cur = head
        # 1、从头到尾遍历链表,把每个数据添加到列表中
        while cur:
            array_list.append(cur.value)
            cur = cur.next

        # 2、对列表进行反转
        array_list.reverse()

        return array_list

if __name__ == '__main__':
    s = Solution()
    # 链表: 100 -> 200 -> 300 -> None
    head = Node(100)
    head.next = Node(200)
    head.next.next = Node(300)

    print(s.get_array_list(head))













