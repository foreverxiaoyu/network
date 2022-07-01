import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# output = torch.reshape(input, (-1, 1, 2, 2))
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Jxy(nn.Module):
    def __init__(self):
        super(Jxy, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output1 = self.relu(input)
        output2 = self.sigmoid(input)
        return output1, output2



jxy = Jxy()

# output1, output2 = jxy(input)
# print(output1)
# print(output2)
writer = SummaryWriter("relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("raw", imgs, step)
    output1, output2 = jxy(imgs)
    writer.add_images("ReLU", output1, step)
    writer.add_images("Sigmoid", output2, step)
    step = step + 1

writer.close()
