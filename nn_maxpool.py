import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)

from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)
# input = torch.reshape(input, (-1, 1, 5, 5))


class Jxy(nn.Module):

    def __init__(self):
        super(Jxy, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)     # 这里通道数就没有变，所以不要reshape

    def forward(self, input):
        output = self.maxpool1(input)
        return output


jxy = Jxy()

writer = SummaryWriter("maxpool")
step = 0

for data in dataloader:
    imgs, target = data
    writer.add_images("input", imgs, step)
    output = jxy(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
