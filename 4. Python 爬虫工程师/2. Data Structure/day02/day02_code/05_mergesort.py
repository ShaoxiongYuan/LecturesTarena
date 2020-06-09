"""
归并排序
思路：
    1、先拆分 : 一直拆到每个小列表中只有1个元素为止
    2、再合并 : 从内到外,依次排序合并
"""
# li: [6, 5, 3, 8]
def merge_sort(li):
    # 递归出口
    if len(li) == 1:
        return li

    # 1、先拆分
    mid = len(li) // 2
    left = li[:mid]
    right = li[mid:]
    # 递归思想:
    # merge_sort([6,5,3,1])
    # merge_sort([8,7,2,4]) ... 依次拆分
    left_li = merge_sort(left)
    right_li = merge_sort(right)

    # 2、再合并
    return merge(left_li, right_li)


def merge(left_li, right_li):
    """合并功能函数"""
    result = []
    # [5,6]   [1,3]
    while len(left_li)>0 and len(right_li)>0:
        if left_li[0] >= right_li[0]:
            result.append(right_li.pop(0))
        else:
            result.append(left_li.pop(0))
    # 循环结束时,一定有一个列表为空
    result += left_li
    result += right_li

    return result


if __name__ == '__main__':
    # 归并排序生成的是排序后的新列表,原列表不变
    li = [6, 5, 3, 1, 666, 8, 8, 7, 2, 4]
    print(merge_sort(li))
    print(li)








