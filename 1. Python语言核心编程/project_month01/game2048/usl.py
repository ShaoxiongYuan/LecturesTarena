from bll import GameCoreController

class GameConsoleView:
    def __init__(self):
        self.__controller = GameCoreController()

    def __start(self):
        self.__controller.generate_new_number()
        self.__controller.generate_new_number()
        self.__print_map()

    def __print_map(self):
        for r in range(4):
            for c in range(4):
                print(self.__controller.map[r][c],end = "\t")
            print()

    def __update(self):
        self.__move_map_for_input()
        self.__controller.generate_new_number()
        self.__print_map()
        if self.__controller.is_game_over():
            print("游戏结束")

    def __move_map_for_input(self):
        dir = input("请输入移动方向：")
        if dir == "w":
            self.__controller.move_up()
        elif dir == "s":
            self.__controller.move_down()
        elif dir == "a":
            self.__controller.move_left()
        elif dir == "d":
            self.__controller.move_right()

    def main(self):
        self.__start()
        while True:
            self.__update()