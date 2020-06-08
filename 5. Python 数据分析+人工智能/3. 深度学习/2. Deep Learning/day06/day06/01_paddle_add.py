# 01_paddle_add.py
import paddle.fluid as fluid

# 定义两个常量(张量)
x = fluid.layers.fill_constant(shape=[1],
                               dtype="int64",
                               value=5)
y = fluid.layers.fill_constant(shape=[1],
                               dtype="int64",
                               value=1)
z = x + y  # 执行张量相加(操作)

# 定义执行器
place = fluid.CPUPlace()  # 指定在CPU上运行
exe = fluid.Executor(place)  # 创建执行器
# paddle在执行时,只需要指定执行哪个program
# 会执行该program下所有的op, 所以不需要指定执行哪个op
result = exe.run(fluid.default_main_program(),
                 fetch_list=[z]) # 返回哪些结果
print(result[0]) # result是多维数组
