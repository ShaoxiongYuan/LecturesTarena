# **Day02回顾**

- **Hadoop核心组件**

  ```
  【1】分布式文件系统 - HDFS
  	1.1) 进程一: NameNode
  	1.2) 进程二: DataNode
  	1.3) 进程三: SecondaryNameNode
  
  【2】分布式计算系统 - MapReduce
  
  【3】分布式资源管理 - Yarn
  	3.1) 进程一: ResourceManager
  	3.2) 进程二: NodeManager
  ```

- **MapReduce过程**

  ```
  【1】Map
  【2】Shuffle & Sort
  【3】Reduce
  
  【4】Hadoop分布式计算流程
  	4.1) 对输入的文件进行切分
  	4.2) 将切分的数据做map处理(将数据分类,数据(k,v)键值对数据)
  	4.3) Shuffle & Sort,将相同的数据放在一起,并对数据进行排序(key)
  	4.4) 对map的数据进行统计计算(reduce)
  	4.5) 格式化输出数据
  ```

- **MapReduce代码实现过程**

  ```python
  from mrjob.job import MRJob
  
  class Xxx(MRJob):
  	def mapper(self, _, line):
          # 参数一: 每行行首偏移量
          # 参数二: 每行内容(字符串)
          pass
      
      def reducer(self, key, values):
          # 参数一: mapper()+shuffle+sort之后的那些key们
          # 参数二: 每个key进行map之后的序列
          pass
      
  if __name__ == '__main__':
      Xxx.run()
  ```

- **MapReduce程序的运行方式**

  ```
  【1】本地模式(-r local)
  	python3 xxx.py -r local 文件名
  【2】Hadoop模式(-r hadoop)
  	python3 xxx.py -r hadoop 文件名
  ```

- **Hive必须记住**

  ```
  【1】Hive不是数据库,只是一个工具(大数据离线分析的工具)
  【2】本质原理: 将HQL语句转为MapReduce任务执行(分布式计算)
  【3】Hive优点
  	3.1) 学习成本低,只要会SQL
  	3.2) 开发效率高
  	3.3) 海量数据高性能分布式查询和分析
  【4】Hive缺点
  	4.1) 不支持在线处理
  	4.2) 不支持行级别的增删改
  【5】Hive应用场景
  	不适合低延迟的应用
  	适合静态离线批处理的Hadoop之上
  ```

- **Hive数据模型**

  ```
  【1】数据库 - database
  	创建库会对应到HDFS的一个目录
  	默认路径: /user/hive/warehouse/库名.db
  【2】表 - table
  	创建表也会对应到HDFS的一个目录
  	默认路径: /user/hive/warehouse/库名.db/表名
  【3】把文件放到 表名 对应的目录中
  	3.1) 方法一
  		hadoop fs -put 文件 /user/hive/warehouse/......
  	3.2) 方法二
  		load data local inpath '文件' into table 表名;
  ```

- **Hive本质**

  ```
  【1】把HDFS上的分布式存储的文件,映射为一张数据库的表
  【2】MySQL数据库的hive库中,存储了元数据的信息
  ```

- **HQL特殊**

  ```
  【1】创建表时,指定行和列分隔符
  	行: row format delimited
  	列: fields terminated by '分隔符'
  【2】数据类型字符串为: string
  【3】通过加载本地文件到指定表中
  	3.1) 方法一
  		hadoop fs -put 文件 /user/hive/warehouse/......
  	3.2) 方法二
  		load data local inpath '文件' into table 表名;
  ```

  























