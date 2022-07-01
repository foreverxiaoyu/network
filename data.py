from torch.utils.data import Dataset
from PIL import Image
import os

root_dir = "/home/dell/learning/data/flower_photos"
roses_label_dir = "roses"
tulips_label_dir = "tulips"

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path) #from torch.utils.data import Dataset
from PIL import Image
import os

root_dir = "/home/dell/learning/data/flower_photos"
label_dir = "roses"

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path) #用了PIL中的打开文件#
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

rose = MyData(root_dir, roses_label_dir)
tulips = MyData(root_dir, tulips_label_dir)
train_dataset = rose + tulips
index = int(input("please input index"))
img1, label1 = rose[index]
img2, label2 = tulips[index]
img1.show()
img2.show()
print(len(rose))
print(len(tulips))
print(len(train_dataset))

