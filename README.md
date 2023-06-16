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
python train.py -a resnet [后面是需要修改的参数，不添加就是默认值]
```

# 改进思路
## 1.激活函数
注意，在训练baseline时，7x7Conv卷积操作后有一个激活函数，可进行调整

使用方法：
* 选择测试的激活函数集：在train.py中`elif args.task=="activation":`下方`act_func=['sigmoid','relu','gelu','leaky-relu']`中修改（默认即可）
* 在命令行加上 `[--task activation]`(将任务调整为测试激活函数，默认为none，表示原来的训练任务)

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
* 在你开始进行第二部分的测试前，需要把你在第一部分测试得到的最优激活函数放到triplet.py头部的bestrelu变量中
* 然后在命令行加上`[--task sigmoid]`执行该任务，其他参数可自行调整

## 3.修改通道池化层的池化方式
原论文中使用的方法是获取通道的Max与Avg，但是还可以获取通道的其他特征量，如中位数、l1范数、l2范数

在你开始第三部分测试前，需要将第二部分得到的最优激活函数放入triplet.py头部bestsigmoid中

使用方法：

修改triplet.py中TripletAttention类的init（）中的参数pool_types ，可自定义组合：
* 原组合：`pool_types=["avg", "max"]`
* 推荐组合1：`pool_types=["avg", "max","median"]`
* 推荐组合2：`pool_types=["l1", "max"]`
* 推荐组合3：`pool_types=["l2", "max"]`
* 测试推荐组合后，可根据测试结果在l1和l2和avg中选择最优[t1]，在"max"+"median"和“max”中选择最优[t2]
* 后续组合4：`pool_types=[t1,t2]`

注1：与上面两个任务不同，第三个任务的不同组合需要一个一个配置，即每次都要修改pool_types
注2：不需要添加 `[--task sigmoid]`命令，使用默认的none即可