import argparse
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from resnet import *
#from van import *
from PIL import ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

import wandb
#等实现triplet后再拓展


parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
# parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet",
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)

parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

#resnet网络默认架构resnet18
parser.add_argument("--depth", default=18, type=int, metavar="D", help="model depth")

parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=666,
    metavar="BS",
    help="Selected Seed (default: 666)",
)

#训练不同网络在这里修改前缀名：
parser.add_argument(
    "--prefix",
    type=str,
    default="tripletnet",
    metavar="PFX",
    help="prefix for logging & checkpoint saving",
)
parser.add_argument(
    "--evaluate", dest="evaluate", action="store_true", help="evaluation only"
)

#使用Triplet在这里设置：
parser.add_argument("--att-type", type=str, choices=["TripletAttention","VAN","TripletCA","CA","TripletLKA"], default="TripletAttention")

#任务类型，用于统一wandb的数据管理
parser.add_argument("--task",type=str,choices=["Triplet_activation","Triplet_sigmoid","Triplet_pool","baseline"])


#设置activation
parser.add_argument("--activation",type=str,choices=['sigmoid','relu','gelu','leaky-relu'],default="sigmoid")

#设置sigmoid
parser.add_argument("--sigmoid",type=str,choices=["softmax","sigmoid","tanh"],default="sigmoid")

#设置Triplet CA的消融实验

#设置pool
#0:["avg", "max"]
#1:["avg", "max","median"]
#2:["l1", "max"]
#3:["l2", "max"]
#4:[待定]
parser.add_argument("--pool",type=int,default=0)
best_prec1 = 0

if not os.path.exists("./checkpoints"):
    os.mkdir("./checkpoints")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print("args", args)

    #日志名称在这里设置：,'relu','gelu','leaky-relu'
    if args.task=="Triplet_activation":
        m_name=args.activation
    elif args.task=="Triplet_sigmoid":
        m_name=args.sigmoid
    elif args.task=="Triplet_pool":
        if args.pool==0:
            m_name="[avg, max]"
        elif args.pool==1:
            m_name="[avg, max,median]"
        elif args.pool==2:
            m_name="[l1, max]"
        elif args.pool==3:
            m_name="[l2,max]"
        else:
            m_name="[avg, max]"
    elif args.task=="baseline":
        if args.att_type=="TripletCA":
            m_name = "+TripletCA"
        elif args.att_type == "TripletLKA":
            m_name = "+TripletLKA"
        elif args.att_type=="TripletAttention":
            m_name="+Triplet"
        elif args.att_type=="CA":
            m_name = "+CA"
        elif args.att_type=="None":
            m_name="no attention"
    else:
        m_name="fault"



    wandb.init(project=args.task,name=m_name)

    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    #设置数据集
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = datasets.CIFAR100('./datasets', train=True,
                                             download=True, transform=transform)
    test_set = datasets.CIFAR100('./datasets', train=False,
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers)

    # create model
    #可以选择不同版本的网络
    if args.arch == "resnet":
        model = ResidualNet("CIFAR100",
                            depth=args.depth,
                            num_classes=1000,
                            att_type=args.att_type,
                            activation=args.activation,
                            replace_sigmoid=args.sigmoid,
                            pool_type=args.pool)
    elif args.arch == "VAN":
        model = van_b0()

    model.to(device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # model = torch.nn.DataParallel(model).cuda()
    wandb.watch(model)
    print ("model")
    print (model)

    """获取模型参数个数"""
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    wandb.log({"parameters": sum([p.data.nelement() for p in model.parameters()])})
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # import pdb
    # pdb.set_trace()

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)[0]

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.prefix,
        )
    # if args.task=="none":
    #     for epoch in range(args.start_epoch, args.epochs):
    #         adjust_learning_rate(optimizer, epoch)
    #
    #         # train for one epoch
    #         train(train_loader, model, criterion, optimizer, epoch)
    #
    #         # evaluate on validation set
    #         prec1 = validate(val_loader, model, criterion, epoch)[0]
    #
    #         # remember best prec@1 and save checkpoint
    #         is_best = prec1 > best_prec1
    #         best_prec1 = max(prec1, best_prec1)
    #         save_checkpoint(
    #             {
    #                 "epoch": epoch + 1,
    #                 "arch": args.arch,
    #                 "state_dict": model.state_dict(),
    #                 "best_prec1": best_prec1,
    #                 "optimizer": optimizer.state_dict(),
    #             },
    #             is_best,
    #             args.prefix,
    #         )
    # elif args.task=="activation":
    #     #测试多种激活函数
    #     act_func=['sigmoid','relu','gelu','leaky-relu']
    #     #act_func = ['gelu', 'leaky-relu']
    #     acc_results={}
    #     loss_results={}
    #     for activation in act_func:
    #         print("\nTraining with {0} activation function\n",activation)
    #         # 先定义模型、优化器、
    #         if args.arch == "resnet":
    #             act_model = ResidualNet("CIFAR100", args.depth, 1000, args.att_type, activation=activation)
    #         elif args.arch == "VAN":
    #             act_model = van_b0()
    #
    #         act_model=act_model.to(device)
    #         act_criterion = nn.CrossEntropyLoss().to(device)
    #
    #         act_optimizer = torch.optim.SGD(
    #             act_model.parameters(),
    #             args.lr,
    #             momentum=args.momentum,
    #             weight_decay=args.weight_decay,
    #         )
    #         acc_result=[]
    #         loss_result=[]
    #         for epoch in range(args.start_epoch, args.epochs):
    #             adjust_learning_rate(act_optimizer, epoch)
    #
    #             # train for one epoch
    #             train(train_loader, act_model, act_criterion, act_optimizer, epoch)
    #
    #             top1,top5,loss=validate(val_loader, act_model, act_criterion, epoch)
    #             acc_result.append(top1)
    #             loss_result.append(loss)
    #         acc_results[activation]=acc_result
    #         loss_results[activation]=loss_result
    #     #绘图
    #     multi_draw(args.task,act_func,acc_results,loss_results)
    #
    # elif args.task=="sigmoid":
    #     test_func=["softmax","sigmoid","tanh"]
    #     acc_results = {}
    #     loss_results = {}
    #     for activation in test_func:
    #         print("\nTraining with {0} activation function\n", activation)
    #         # 先定义模型、优化器、
    #         if args.arch == "resnet":
    #             act_model = ResidualNet("CIFAR100", args.depth, 1000, args.att_type, replace_sigmoid=activation)
    #         elif args.arch == "VAN":
    #             act_model = van_b0()
    #
    #         act_model=act_model.to(device)
    #         act_criterion = nn.CrossEntropyLoss().to(device)
    #
    #         act_optimizer = torch.optim.SGD(
    #             act_model.parameters(),
    #             args.lr,
    #             momentum=args.momentum,
    #             weight_decay=args.weight_decay,
    #         )
    #         acc_result = []
    #         loss_result = []
    #         for epoch in range(args.start_epoch, args.epochs):
    #             adjust_learning_rate(act_optimizer, epoch)
    #
    #             # train for one epoch
    #             train(train_loader, act_model, act_criterion, act_optimizer, epoch)
    #
    #             top1, top5, loss = validate(val_loader, act_model, act_criterion, epoch)
    #             acc_result.append(top1)
    #             loss_result.append(loss)
    #         acc_results[activation] = acc_result
    #         loss_results[activation] = loss_result
    #     multi_draw(args.task, test_func, acc_results, loss_results)






def multi_draw(task,func,acc,loss):
    x = np.linspace(1, args.epochs, args.epochs)
    # 准确度绘图
    plt.figure(figsize=(10, 10))
    for activation in func:
        plt.plot(x, acc[activation])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Accuracy(top1)")
    plt.xlabel("epoch")
    plt.ylabel("Test accuracy")
    plt.legend(func)
    plt.savefig(task+"Accuracy")
    plt.show()

    # 损失绘图
    plt.figure(figsize=(10, 10))
    for activation in func:
        plt.plot(x, loss[activation])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("Test Loss")
    plt.legend(func)
    plt.savefig(task+"Loss")
    plt.show()
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    # log stats to wandb
    wandb.log(
        {
            "epoch": epoch,
            "Top-1 accuracy": top1.avg,
            "Top-5 accuracy": top5.avg,
            "loss": losses.avg,
        }
    )

    return top1.avg,top5.avg,losses.avg


def save_checkpoint(state, is_best, prefix):
    filename = "./checkpoints/%s_checkpoint.pth.tar" % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "./checkpoints/%s_model_best.pth.tar" % prefix)
        wandb.save(filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    wandb.log({"lr": lr})


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.reshape((batch_size, -1))  # Reformat the output of topk to match the shape of input tensor
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()