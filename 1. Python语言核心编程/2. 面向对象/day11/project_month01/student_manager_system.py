"""
    学生信息管理系统
"""


class StudentModel:
    """
        学生模型
    """

    def __init__(self, name="", age=0, sex="", score=0, id=0):
        self.name = name
        self.age = age
        self.sex = sex
        self.score = score
        self.id = id


class StudentManagerView:
    """
        界面视图(逻辑)
    """

    def __init__(self):
        self.__manager = StudentManagerController()

    def __display_menu(self):
        print("1)添加学生信息")
        print("2)显示学生信息")
        print("3)删除学生信息")
        print("4)修改学生信息")

    def __select_menu(self):
        item = input("请输入选项：")
        if item == "1":
            self.__input_students()
        elif item == "2":
            self.__output_students()

    def main(self):
        while True:
            self.__display_menu()
            self.__select_menu()

    def __input_students(self):
        name = input("请输入姓名：")
        sex = input("请输入性别：")
        age = int(input("请输入年龄："))
        score = int(input("请输入成绩："))
        stu = StudentModel(name, age, sex, score)
        # ....
        self.__manager.add_student(stu)

    def __output_students(self):
        pass


class StudentManagerController:
    """
        核心逻辑(控制)
    """

    def __init__(self):
        self.__list_stu = []

    def add_student(self, stu_target):
        # 1. 为学生创建id
        # 2. 将学生加入到列表中__list_stu
        pass


view = StudentManagerView()
view.main()
