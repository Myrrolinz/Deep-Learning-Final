import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from model import resnet50
import os
import torch.optim as optim
from tqdm import tqdm


epoches = 5  # 训练次数
batch_size = 1024  # 训练批次(一次训练的数据
CIFAR100_class = 100  # 数据集的分类类别数量
learning_rate = 0.002  # 模型学习率


def read_data():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_val = transforms.Compose([
         transforms.Resize((32, 32)),
         # transforms.Resize(256),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True,
                                                  download=True, transform=transform_train)

    val_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
                                                download=True, transform=transform_val)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    return train_dataset, val_dataset, train_loader, val_loader



def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据读取
    train_dataset, val_dataset, train_loader, val_loader = read_data()

    # 模型加载
    # model = res2net50(num_classes=CIFAR100_class)
    # model = res2net50_1(num_classes=CIFAR100_class)
    model = resnet50(num_classes=CIFAR100_class)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc =  0.0
    save_model_path = './res/resnet50_best_model.pth'

    for epoch in range(epoches):
        # 训练
        print("start training......")
        model.train()
        running_loss_train = 0.0
        train_accurate = 0.0
        train_bar = tqdm(train_loader)
        for images, labels in train_bar:
            optimizer.zero_grad()

            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, dim=1)[1]
            train_accurate += torch.eq(predict, labels.to(device)).sum().item()
            running_loss_train += loss.item()

        train_accurate = train_accurate / len(train_dataset)
        running_loss_train = running_loss_train / len(train_dataset)

        print('[epoch %d/%d] train_loss: %.7f  train_accuracy: %.3f' %
              (epoch + 1, epoches, running_loss_train, train_accurate))

        # 验证
        print("start validating......")
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_loader = tqdm(val_loader)
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(val_dataset)

        print('[epoch %d/%d] train_loss: %.7f  val_accuracy: %.3f' %
              (epoch + 1, epoches, running_loss_train, val_accurate))

        # 选择最best的模型进行保存 评价指标此处是acc
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    train()