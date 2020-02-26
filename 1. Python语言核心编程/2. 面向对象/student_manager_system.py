"""
    学生信息管理系统
    1. 显示学生信息
        -- 在controller类中提供只读属性(列表)
        -- 在view类中遍历只读属性显示信息(列表)
    2. 修改学生信息
        -- 在controller类中提供修改信息算法
        -- 在view类中定义修改信息的界面逻辑
    3. 删除学生信息
        -- 在controller类中提供删除算法
        -- 在view类中定义删除信息的界面逻辑( id )
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
        age = int(input("请输入年龄："))
        score = int(input("请输入成绩："))
        stu = StudentModel(name, age, sex, score)
        self.__manager.add_student(stu)

    def __output_students(self):
        for item in self.__manager.list_stu:
            print("%s的性别是%s,年龄是%d,成绩是%f" % (item.name, item.sex, item.age, item.score))

    def __modify_student(self):
        stu = StudentModel()
        stu.id = int(input("请输入需要修改的学生编号："))
        stu.name = input("请输入需要修改的学生姓名：")
        stu.sex = input("请输入需要修改的学生性别：")
        stu.age = int(input("请输入需要修改的学生年龄："))
        stu.score = int(input("请输入需要修改的学生成绩："))
        if self.__manager.update_student(stu):
            print("修改成功")
        else:
            print("修改失败")

    def __delete_student(self):
        id = int(input("请输入需要删除的学生编号："))
        if self.__manager.remove_student(id):
            print("删除成功")
        else:
            print("删除失败")


class StudentManagerController:
    """
        核心逻辑(控制)
    """

    # 类变量：初始编号
    __init_id = 1000

    @classmethod
    def __generate_id(cls, stu):
        stu.id = cls.__init_id
        cls.__init_id += 1

    def __init__(self):
        self.__list_stu = []

    @property
    def list_stu(self):
        return self.__list_stu

    def add_student(self, stu_target):
        """
            添加新学生
        :param stu_target:需要添加的学生
        """
        StudentManagerController.__generate_id(stu_target)
        self.__list_stu.append(stu_target)

    def update_student(self, stu):
        """
            修改学生信息
        :param stu: 需要修改的信息
        :return: 是否修改成功
        """
        for item in self.__list_stu:
            if item.id == stu.id:
                item.name = stu.name
                item.age = stu.age
                item.score = stu.score
                item.sex = stu.sex
                return True
        return False

    def remove_student(self, id):
        """
            移除学生
        :param id:学生编号
        :return:移除是否成功
        """
        for item in self.__list_stu:
            if item.id == id:
                self.__list_stu.remove(item)
                return True
        return False


view = StudentManagerView()
view.main()
