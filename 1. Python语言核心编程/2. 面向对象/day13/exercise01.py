"""
    一家公司有如下几种岗位:
        程序员：底薪 + 项目分红
        测试员：底薪 + Bug * 5
        ....
    创建员工管理器
        记录所有员工
        提供计算总工资的功能
    叙述：
        三大特征：
            封装：根据需求划分为员工管理器、程序员、测试员
            继承：创建员工类,隔离员工管理器与具体员工的变化
            多态：员工管理器调用员工,具体员工重写员工的计算薪资方法,添加具体员工对象
                    (调父)             (重写)                 (创建子类对象)
                 总结多态步骤：面对变化点，需要使用3步进行处理。
                 价值(作用)：程序灵活(扩展性强 = 开闭原则)
        六大原则：
            开闭原则：增加员工种类,员工管理器不变.
            单一职责：员工管理器：统一管理所有员工(薪资...)
                    程序员：定义该员工的薪资算法(底薪 + 项目分红)
                    测试员：定义该员工的薪资算法(底薪 + Bug * 5)
            依赖倒置：员工管理器调用员工,不调用程序员/测试员
            组合复用：员工管理器通过组合关系,使用各种员工的薪资算法.
            里氏替换：向员工管理器添加的是员工子类对象
                    员工子类先调用员工类计算薪资方法返回底薪,再添加自己的薪资逻辑.
            迪米特：每个具体员工类之间低耦合
                   员工管理器与具体员工之间低耦合


"""

class EmployeeManager:
    def __init__(self):
        self.__list_employees = []

    def add_employee(self, emp):
        self.__list_employees.append(emp)

    def get_total_salary(self):
        total_salary = 0
        for item in self.__list_employees:
            # 调用父类
            total_salary += item.calculate_salary()
        return total_salary

class Employee:
    def __init__(self, base_salary=0):
        self.base_salary = base_salary

    def calculate_salary(self):
        return self.base_salary

# --------------------------
class Programmer(Employee):
    def __init__(self, base_salary=0, bonus=0):
        super().__init__(base_salary)
        self.bonus = bonus

    # 重写
    def calculate_salary(self):
        return super().calculate_salary() + self.bonus

class Tester(Employee):
    def __init__(self, base_salary=0, bug_count=0):
        super().__init__(base_salary)
        self.bug_count = bug_count

    def calculate_salary(self):
        return super().calculate_salary() + self.bug_count * 5


manager = EmployeeManager()
# 创建子类对象
manager.add_employee(Programmer(8000, 20000))
manager.add_employee(Tester(5000, 100))
print(manager.get_total_salary())
