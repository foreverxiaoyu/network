import torch

# 方式一的加载
import torchvision.models
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear

model = torch.load("vgg16_method1.pth")
print(model)
# 方式二
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

print(vgg16)

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

# jxy = Jxy()


model = torch.load("jxy_method1.pth")
print(model)
