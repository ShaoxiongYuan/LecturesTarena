# **Day01回顾**

- **大数据特征 - 5v**

  ```
  【1】大体量 - 从TB级别开始算
  【2】多样性 - 数据种类和来源多
  【3】时效性 - 在一定的时间内得到及时处理
  【4】准确性 - 保证一定准确度
  【5】大价值 - 探究深度价值
  ```

- **Hadoop特点**

  ```
  【1】高可靠性
  【2】高扩展性 - 横向扩展
  【3】高效性 - 节点之间动态移动数据
  【4】高容错性 - 副本冗余机制（默认3个副本）
  ```

- **Hadoop核心组件**

  ```
  【1】HDFS - 分布式文件存储系统
  【2】MapReduce - 分布式计算
  【3】Yarn - 资源管理系统
  ```

- **HDFS必须记住**

  ```
  【1】Client
  	1.1) 切分文件(默认128M)
  	1.2) 与NameNode交互获取节点信息以及元数据信息
  	1.3) 与DataNode交互来读取和写入数据
  	
  【2】NameNode(Master)
  	2.1) 存储元数据信息(数据块的映射信息)
  	2.2) 处理所有Client请求
  	2.3) 配置副本策略
  	
  【3】SecondaryNameNode
  	定期同步NameNode的元数据以及日志信息
  	
  【4】DataNode
  	4.1) 数据存储节点
  	4.2) 汇报存储信息给NameNode,NameNode更新元数据信息
  	
  【5】相关进程 - jps
  	5.1) NameNode
  	5.2) DataNode
  	5.3) SecondaryNameNode
  	
  【6】HDFS写文件流程
  	6.1) 客户端拆分文件为128M的块,并通知NameNode
  	6.2) NameNode寻找DataNode,并返给客户端
  	6.3) 客户端直接找到相关的DataNode,对块进行写入
  	6.4) DataNode进行流水线的复制
  	6.5) NameNode来更新元数据
  	
  【7】HDFS读文件流程
  	7.1) 客户端向NameNode发出读请求
  	7.2) NameNode查询元数据,并交给客户端
  	7.3) 客户端会根据元数据信息,在对应的DataNode上直接获取数据
  ```

- **MapReduce必须记住**

  ```
  【1】Hadoop1.0中的 MapReduce
  	1.1) JobTracker
  		 接收任务
  		 资源调度
  		 监控任务的执行状态
  	1.2) TaskTracker
  		 具体计算任务并向JobTracker汇报任务状态
  		 
  【2】Hadoop2.0中的 MapReduce
  	将JobTracker中资源调度的工作独立出来,即Yarn
  ```

- **JDK安装步骤**

  ```
  【1】下载解压
  【2】添加环境变量
  【3】刷新环境变量
  【4】验证
  ```

- **Hadoop安装步骤**

  ```
  【1】下载解压到指定路径
  【2】添加JAVA环境变量(/usr/local/hadoop2.10/etc/hadoop/.sh)
  【3】添加Hadoop环境变量并刷新(.bashrc文件)
  【4】配置免密登陆
  【5】配置伪分布式(...../etc/hadoop/xxx.xml)
  【6】配置Yarn
  【7】格式化namenode
  【8】启动hadoop所有组件
  	路径: /usr/local/hadoop2.10/sbin/
  	启动文件: ./start-all.sh
  	
  终端jps
  	HDFS进程1: NameNode
  	HDFS进程2: DataNode
  	HDFS进程3: SecondaryNameNode
  	
  	Yarn进程1: ResourceManager
  	Yarn进程2: NodeManager
  ```

  











