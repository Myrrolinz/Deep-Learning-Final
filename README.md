# Deep-Learning-Final
NKU 深度学习 2023春季大作业

## wandb使用方法
在调用main.py前，需要登录wandb。登录流程如下：
1. 如果没有下载,在命令行执行`pip install wandb`
2. 在命令行执行`wandb login`
3. 点击第二个连接，将获得的密码输入命令行

## 使用train.py
1. 直接调用main()
2. 在命令行输入
```
python train.py -a resnet --depth 50 后面是需要修改的参数，不添加就是默认值]
```
## 参数管理
1. `--arch`:选择架构 (resnet/van)
2. `--depth`:选择模型深度   使用`--depth 50`
3. `--epochs`:训练周期 使用`--epochs 50`,到后期使用`--epochs 100`
4. `--att-type`:注意力模型 
   1. "TripletAttention"
   2. "VAN",
   3. "TripletCA",
   4. "CA",
   5. "TripletLKA"
5. `--task`:训练任务

# 改进思路
## 1.激活函数
注意，在训练baseline时，7x7Conv卷积操作后有一个激活函数，可进行调整
使用方法： 
* 默认`--depth 50 --epochs 50 --att-type TripletAttention --task Triplet_activation `
* 调整`--activation ('sigmoid','relu','gelu','leaky-relu')`

[//]: # (* 选择测试的激活函数集：在train.py中`elif args.task=="activation":`下方`act_func=['sigmoid','relu','gelu','leaky-relu']`中修改（默认即可）)

[//]: # (* 在命令行加上 `[--task activation]`&#40;将任务调整为测试激活函数，默认为none，表示原来的训练任务&#41;)

一个小demo:

1. 使用gelu:
* epoch1： Prec@1 22.050 Prec@5 52.140
* epoch2: Prec@1 35.260 Prec@5 67.530

2. 使用leaky-relu:
* epoch1: Prec@1 24.480 Prec@5 53.360
* epoch2: Prec@1 34.320 Prec@5 66.470

输出图像将所有激活函数的acc(用的是top1)放在一张图，loss放在另一张图。
## 2.sigmoid激活函数调整
与1不同，这个激活函数是在正则化后的激活函数，原论文中使用的是sigmoid

sigmoid对比函数有：tanh、softmax

测试方式：
* 默认`--depth 50 --epochs 50 --att-type TripletAttention --task Triplet_sigmoid`
* 使用task1中性能优秀的activation`[--activation 最好的那个]`
* 调整`[--sigmoid ("softmax","sigmoid","tanh")]`

[//]: # (* 在你开始进行第二部分的测试前，需要把你在第一部分测试得到的最优激活函数放到triplet.py头部的bestrelu变量中)

[//]: # (* 然后在命令行加上`[--task sigmoid]`执行该任务，其他参数可自行调整)

## 3.修改通道池化层的池化方式
原论文中使用的方法是获取通道的Max与Avg，但是还可以获取通道的其他特征量，如中位数、l1范数、l2范数

在你开始第三部分测试前，需要将第二部分得到的最优激活函数放入triplet.py头部bestsigmoid中


修改triplet.py中TripletAttention类的init（）中的参数pool_types ，可自定义组合：
* 原组合：`pool_types=["avg", "max"]`
* 推荐组合1：`pool_types=["avg", "max","median"]`
* 推荐组合2：`pool_types=["l1", "max"]`
* 推荐组合3：`pool_types=["l2", "max"]`
* 测试推荐组合后，可根据测试结果在l1和l2和avg中选择最优[t1]，在"max"+"median"和“max”中选择最优[t2]
* 后续组合4：`pool_types=[t1,t2]`

使用方法：
* 默认`--depth 50 --epochs 50 --att-type TripletAttention --task Triplet_pool`
* 使用task1和task2最好的组合`[--activation 最好的 --sigmoid 最好的]`
* 调整`[--pool (0、1、2、3)]`
* 后续组合需要另外修改trilpet代码

注1：与上面两个任务不同，第三个任务的不同组合需要一个一个配置，即每次都要修改pool_types

注2：不需要添加 `[--task sigmoid]`命令，使用默认的none即可

demo版:epoch=2,pool_types=["avg", "max","median"],bestrelu=relu,bestsigmoid=sigmoid:

1. epoch1:Prec@1 23.110 Prec@5 51.130
2. epoch2:

直观感受：relu相比leaky-relu和gelu快很多

## coordinate与triplet结合
coordinate的意义在于：在获取通道特征的时候能够结合空间特征，即通过将空间拆分成两个方向的向量，分别获得该方向的通道特征(CxWx1、Cx1xH)

triplet的意义在于：讨论了空间与通道之间的交互关系，分别获得了每个维度的特征图(例：W维-CxHx1)，最后能获得三个维度分别的特征图

两者相同之处：其Avg池化均为1维池化，针对某一维度获取全局特征；

不同之处：最后特征图使用的位置不同。coordinate在最后将特定通道、特定位置的三维权重相乘；triplet先将每个权重图与原矩阵相乘，再求三者的平均

个人认为，triplet的优越点在于它将C、W和C、H这两个维度获取了交互信息，以往的模型均基于W、H的交互信息，再将其与C相乘

coordinate的优势在于它符合直觉，在获取通道特征时能够有效结合空间特征。

我选择以triplet为骨干，使用coordinate的方法改进C维(WxH)部分的特征图提取，并尝试用这个方法提取其他交互维度的特征图。

对于triplet最后将三个结果求平均的方法，我也想要试着进行改进。

可能从以下方面进行：
1. 完成基础的triplet+CA
2. 在其基础上，添加多种一维池化（参照triplet的z-pool）
3. 最后的Avg采用其他方式，如线性组合/卷积
4. 调整reduction_ratio

使用方法：
* 默认`--depth 50 --epochs 50 --att-type TripletAttention --task TripletCA`
* 使用 task1、2、3中的最佳组合`[--activation (*) --sigmoid (*) --pool (*)]`（还没实现，需要手动修改tripletCA的代码）
* 融合实验：将使用CA的和不使用CA的对比（还没写）
demo
1. 只将C维换成AC:`--att-type TripletCA`
   1. epoch1:Prec@1 24.070 Prec@5 52.220
   2. epoch2:Prec@1 34.570 Prec@5 66.500

## CA：
1. epoch1:19.980 Prec@5 48.360
2. epoch2:Prec@1 29.090 Prec@5 59.640

## Baseline实验
对比Triplet、TripletCA、TripletLKA和CA的性能

参数：
* 默认`--task baseline`
* 调整`-- att-type ()`



