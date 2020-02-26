from bll import StudentManagerController
from model import StudentModel


# 10:20
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
        elif item == "3":
            self.__delete_student()
        elif item == "4":
            self.__modify_student()

    def main(self):
        while True:
            self.__display_menu()
            self.__select_menu()

    def __input_students(self):
        name = input("请输入姓名：")
        sex = input("请输入性别：")
        age = self.__input_number("请输入年龄")
        score = self.__input_number("请输入成绩")
        stu = StudentModel(name, age, sex, score)
        self.__manager.add_student(stu)

    def __input_number(self, message):
        while True:
            try:
                return int(input(message))
            except:
                print("输入有误")

    def __output_students(self):
        for item in self.__manager.list_stu:
            print("%s的性别是%s,年龄是%d,成绩是%f" % (item.name, item.sex, item.age, item.score))

    def __modify_student(self):
        stu = StudentModel()
        stu.id = self.__input_number("请输入需要修改的学生编号：")
        stu.name = input("请输入需要修改的学生姓名：")
        stu.sex = input("请输入需要修改的学生性别：")
        stu.age = self.__input_number("请输入需要修改的学生年龄：")
        stu.score = self.__input_number("请输入需要修改的学生成绩：")
        if self.__manager.update_student(stu):
            print("修改成功")
        else:
            print("修改失败")

    def __delete_student(self):
        id = self.__input_number("请输入需要删除的学生编号：")
        if self.__manager.remove_student(id):
            print("删除成功")
        else:
            print("删除失败")
