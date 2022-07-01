import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Jxy(nn.Module):
    def __init__(self):
        super(Jxy, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # 拿了两个卷积核去扫，所以channel*2

    def forward(self, x):
        x = self.conv1(x)
        return x


jxy = Jxy()

writer = SummaryWriter("conv2d")
stride = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("conv2d_raw", imgs, stride)
    # 64 3 32 32
    output = jxy(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    # 64 6 30 30 -> *** 3 30 30 不改成3(RGB)的话,图片会显示不出来，只能把batch_size*2
    writer.add_images("conv2d_after", output, stride)
    stride = stride + 1

writer.close()
