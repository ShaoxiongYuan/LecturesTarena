"""
Python实现二叉树
"""
class Node:
    """节点类"""
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class Tree:
    def __init__(self):
        """初始化一颗空树"""
        self.root = None

    def add(self, item):
        """在树中添加一个节点"""
        node = Node(item)
        # 空树情况
        if self.root == None:
            self.root = node
            return
        # 不为空树的情况 - 队列思想,从树根开始进行判断
        li = [self.root]
        while li:
            cur = li.pop(0)
            # 判断左孩子
            if not cur.left:
                cur.left = node
                return
            else:
                li.append(cur.left)

            # 判断右孩子
            if not cur.right:
                cur.right = node
                return
            else:
                li.append(cur.right)

    def breadth_travel(self):
        """广度遍历 - 层次遍历, 从上到下,从左到右,利用队列思想,一定是从树根开始"""
        # 空树情况
        if self.root == None:
            return
        # 非空树情况
        li = [self.root]
        while li:
            cur = li.pop(0)
            print(cur.value, end=" ")
            # 添加左孩子
            if cur.left:
                li.append(cur.left)
            # 添加右孩子
            if cur.right:
                li.append(cur.right)
        print()

    def pre_travel(self, root):
        """前序遍历 - 根左右"""
        if not root:
            return

        print(root.value, end=" ")
        self.pre_travel(root.left)
        self.pre_travel(root.right)

    def mid_travel(self, root):
        """中序遍历 - 左根右"""
        if not root:
            return

        self.mid_travel(root.left)
        print(root.value, end=" ")
        self.mid_travel(root.right)

    def last_travel(self, root):
        """后序遍历 - 左右根"""
        if not root:
            return

        self.last_travel(root.left)
        self.last_travel(root.right)
        print(root.value, end=" ")

if __name__ == '__main__':
    t = Tree()
    t.add(1)
    t.add(2)
    t.add(3)
    t.add(4)
    t.add(5)
    t.add(6)
    t.add(7)
    t.add(8)
    t.add(9)
    t.add(10)
    # 广度遍历
    t.breadth_travel()
    # 前序
    t.pre_travel(t.root)
    print()
    # 中序
    t.mid_travel(t.root)
    print()
    # 后序
    t.last_travel(t.root)
    print()








