"""
    游戏核心逻辑模块
"""
import random

from model import Location

class GameCoreController:
    def __init__(self):
        self.__list_merge = None
        # self.__map = [
        #     [2, 0, 0, 2],
        #     [0, 4, 0, 2],
        #     [0, 4, 2, 0],
        #     [2, 0, 2, 0],
        # ]
        self.__map = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        self.__list_empty_location = []

    @property
    def map(self):
        return self.__map

    def __zero_to_end(self):
        """
            0元素移动到末尾
        """
        for i in range(len(self.__list_merge) - 1, -1, -1):
            if self.__list_merge[i] == 0:
                del self.__list_merge[i]
                self.__list_merge.append(0)

    def __merge(self):
        """
            合并相同元素
        """
        self.__zero_to_end()
        for i in range(len(self.__list_merge) - 1):
            if self.__list_merge[i] == self.__list_merge[i + 1]:
                self.__list_merge[i] += self.__list_merge[i + 1]
                del self.__list_merge[i + 1]
                self.__list_merge.append(0)

    def move_left(self):
        """
            向左移动
        """
        for line in self.__map:
            self.__list_merge = line
            self.__merge()

    def move_right(self):
        """
            向右移动
        """
        for line in self.__map:
            self.__list_merge = line[::-1]
            self.__merge()
            line[::-1] = self.__list_merge

    def __square_matrix_transpose(self):
        """
            方阵转置　
        """
        for c in range(len(self.__map) - 1):
            for r in range(c + 1, len(self.__map)):
                self.__map[r][c], self.__map[c][r] = self.__map[c][r], self.__map[r][c]

    def move_up(self):
        """
            向上移动　
        """
        self.__square_matrix_transpose()
        self.move_left()
        self.__square_matrix_transpose()

    def move_down(self):
        """
            向下移动
        """
        self.__square_matrix_transpose()
        self.move_right()
        self.__square_matrix_transpose()

    def generate_new_number(self):
        """
            生成新数字　
        """
        self.__calculate_empty_location()
        if len(self.__list_empty_location) == 0: return
        loc = random.choice(self.__list_empty_location)
        self.__map[loc.r][loc.c] = self.__create_random_number()
        # 将当前空位置在列表中移除
        self.__list_empty_location.remove(loc)

    def __create_random_number(self):
        return 4 if random.randint(1, 10) == 1 else 2

    def __calculate_empty_location(self):
        self.__list_empty_location.clear()
        for r in range(len(self.__map)):
            for c in range(len(self.__map[r])):
                if self.__map[r][c] == 0:
                    # 记录r c
                    # self.__list_empty_location.append((r, c))
                    self.__list_empty_location.append(Location(r, c))

    def is_game_over(self):
        if len(self.__list_empty_location) > 0:
            return False

        # for r in range(4):
        #     for c in range(3):
        #         if self.__map[r][c] == self.__map[r][c + 1]:
        #             return False
        #
        # for c in range(4):
        #     for r in range(3):
        #         if self.__map[r][c] == self.__map[r+1][c]
        #             return False

        for r in range(len(self.__map)):  # 0
            for c in range(len(self.__map[0])-1):  # 0 1 2
                if self.__map[r][c] == self.__map[r][c + 1] or self.__map[c][r] == self.__map[c + 1][r]:
                    return False

        return True

if __name__ == '__main__':
    # 测试
    controller = GameCoreController()
    controller.generate_new_number()
    controller.generate_new_number()
    # controller.move_up()
    print(controller.map)
