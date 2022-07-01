import torch
import torchvision.models
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式一
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式二,保存为字典形式（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Jxy(nn.Module):
    def __init__(self):
        super(Jxy, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            torch.nn.Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


jxy = Jxy()
torch.save(jxy, "jxy_method1.pth")
