# AlgoSelect
## **运行方法**

**运行main**函数后，输入命令

1. **get train**: 生成训练数据集
2. **get test**:生成测试数据集
3. **get result**:训练并评价
4. **get all**:执行以上三步

## 关于项目文件夹

项目使用**IDEA**创建

**dataset**存放数据集（包括生成的数据集）

**lib**存放外部库

**hadoop_in**用于向Hadoop输入数据

**hadoop_out_train**和**hadoop_out_test**用于从Hadoop获取数据，***运行前应删掉这两个文件夹***

## 关于运行环境

**Windows**操作系统

**Java**使用**Java8**

## 关于Hadoop

Hadoop版本为2.7.7

项目里已经包括了Hadoop2.7.7的外部库，以本地模式运行Hadoop应该没有问题。

如果有问题就先在Windows上安装好Hadoop，然后将Hadoop文件夹下面的share/hadoop文件夹移到项目文件夹的lib中，并将其添加为外部库。

## 关于代码结构

**simple**文件夹下的是没有使用Hadoop的代码，**hadoop**文件夹下是使用Hadoop的代码，两者相似度很高。

##### 主要结构如下：

**Main**是主类

**DataFeature**负责生成训练和测试数据集（同时负责调用**HadoopEnhancement**）

**AlgoEvaluator**负责评价训练结果

使用Hadoop的代码还有一个**使用Hadoop的类**：**HadoopEnhancement**



