import torchvision.datasets

# 准备的测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# 将原始数据先进行了PIL到Tensor的转换

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
# batch_size 就是一次抓多少张牌 shuffle (受epoch的影响)就是是否重新排列 drop_last 就是 余数要不要

# 测试数据集中的第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):

    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"Epoch: {epoch}", imgs, step)
        step = step + 1
        # print(imgs.shape)
        # print(targets)
writer.close()
