import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Jxy(nn.Module):
    def __init__(self):
        super(Jxy, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output


jxy = Jxy()

# writer = SummaryWriter("linear")
step = 0
for data in dataloader:
    imgs, target = data
    # writer.add_images("raw", imgs, step)
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = jxy(output)
    print(output.shape)
    # writer.add_images("linear", output, step)
    # step = step + 1

# writer.close()

