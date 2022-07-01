import torch
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
import time

from module import Jxy
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
#
#
# class Jxy(nn.Module):
#     def __init__(self):
#         super(Jxy, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(3, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 32, 5, padding=2),
#             MaxPool2d(2),
#             Conv2d(32, 64, 5, padding=2),
#             MaxPool2d(2),
#             torch.nn.Flatten(),
#             Linear(1024, 64),
#             Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x

jxy = Jxy()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(jxy.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 设置训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print(f"第{i + 1}次的训练开始")
    running_loss = 0
    # 训练步骤开始
    jxy.train()
    for data in train_dataloader:
        imgs, target = data
        outputs = jxy(imgs)
        loss = loss_fn(outputs, target)

        # 优化器模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"第{total_train_step}次的误差为： {loss}\n")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    jxy.eval()
    total_test_loss = 0
    total_accuracy = 0
    nums = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = jxy(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = accuracy + total_accuracy
            nums = nums + 1

    print(f"整体测试集上的Loss： {total_test_loss}")
    print(f"整体测试集上的正确率： {total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(jxy, f"jxy_{i}.pth")

writer.close()
