"""
    技能系统
    叙述
        三大特征：
        六大原则：
"""


class ImpactEffect:
    """
        影响效果,隔离技能释放器与具体效果（眩晕，降低速度...）
    """

    def impact(self):
        pass

class CostSPEffect(ImpactEffect):
    def __init__(self, value=0):
        self.value = value

    def impact(self):
        super().impact()
        print("消耗%d法力" % self.value)

class DamageEffect(ImpactEffect):
    def __init__(self, value=0):
        self.value = value

    def impact(self):
        super().impact()
        print("伤害%d生命" % self.value)

class LowerDeffenseEffect(ImpactEffect):
    def __init__(self, value=0, duration=0):
        self.value = value
        self.duration = duration

    def impact(self):
        super().impact()
        print("降低%f防御力,持续%d秒" % (self.value, self.duration))

class SkillDeployer:
    """
        技能释放器
    """
    def __init__(self, name=""):
        self.name = name
        self.__config_file = self.__load_config_file()
        self.__list_effects = self.__create_impact_effects()

    def __load_config_file(self):
        # 模拟配置文件读取后的数据结构
        return {
            "毁天灭地":["CostSPEffect(200)","DamageEffect(800)"],
            "降龙十八掌":["CostSPEffect(150)","DamageEffect(600)","LowerDeffenseEffect(0.5,10)"],
        }

    def __create_impact_effects(self):
        # 技能名称ｋ ---> 效果名称列表v
        list_effect_names = self.__config_file[ self.name ]
        # 创建子类对象()
        # list_effect_objects = []
        # for itme in list_effect_names:
        #     effect_obj = eval(itme)
        #     list_effect_objects.append(effect_obj)
        # return list_effect_objects
        return [eval(item) for item in list_effect_names]

    def deploy_skill(self):
        print(self.name,"释放啦")
        for item in self.__list_effects:
            # 调用父类(影响效果)
            item.impact()

xlsbz = SkillDeployer("降龙十八掌")
xlsbz.deploy_skill()
xlsbz.deploy_skill()



