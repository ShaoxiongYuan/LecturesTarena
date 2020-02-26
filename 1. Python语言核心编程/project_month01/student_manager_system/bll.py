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