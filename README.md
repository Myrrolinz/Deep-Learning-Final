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
注意，在训练baseline时，7x7Conv卷积时未使用激活函数，即论文中的sigmoid

使用方法：
* 选择测试的激活函数集：在train.py中'elif args.task=="activation":'下方'act_func=['sigmoid','relu','gelu','leaky-relu']'中修改（默认即可）
* 在命令行加上 '[--task activation]'(将任务调整为测试激活函数，默认为none，表示原来的训练任务)

一个小demo:

1. 使用gelu:
* epoch1： Prec@1 22.050 Prec@5 52.140
* epoch2: Prec@1 35.260 Prec@5 67.530

2. 使用leaky-relu:
* epoch1: Prec@1 24.480 Prec@5 53.360
* epoch2: Prec@1 34.320 Prec@5 66.470

输出图像将所有激活函数的acc(用的是top1)放在一张图，loss放在另一张图。

