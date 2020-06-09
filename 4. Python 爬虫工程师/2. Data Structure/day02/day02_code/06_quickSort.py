"""
Python实现快速排序
思路：
    1、left遇到比基准值大的,停！
    2、right遇到比基准值小的,停！
    3、交换位置
    4、当第一次left>right时,right的值即为基准值的正确位置, 交换基准值和right下标对应的值
"""
# first的值就为基准值的下标索引
def quick_sort(li, first, last):
    # 递归出口
    if first > last:
        return

    # 找到基准值正确位置的代码
    mid_index = get_index(li, first, last)
    # 递归思想
    quick_sort(li, first, mid_index-1)
    quick_sort(li, mid_index+1, last)

def get_index(li, first, last):
    """获取基准值正确位置的下标索引"""
    mid = li[first]
    lcur = first + 1
    rcur = last

    sign = False
    while not sign:
        # 左游标右移,遇到比基准值大的,停！
        while lcur <= rcur and li[lcur] <= mid:
            lcur += 1
        # 右游标左移,遇到比基准值小的,停！
        while lcur <= rcur and li[rcur] >= mid:
            rcur -= 1

        # 当左游标>右游标时,把基准值和右游标互换
        if lcur > rcur:
            sign = True
            li[first],li[rcur] = li[rcur],li[first]
        else:
            li[lcur],li[rcur] = li[rcur],li[lcur]

    return rcur

if __name__ == '__main__':
    li = [6, 5, 3, 1, 2, 4, 8, 9, 666, 333]
    quick_sort(li, 0, len(li)-1)
    print(li)











