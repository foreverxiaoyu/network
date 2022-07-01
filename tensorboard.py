from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

image_path = "/home/dell/learning/data/flower_photos/tulips/11746080_963537acdc.jpg"
img = Image.open(image_path)
img_array =np.array(img)
print(img_array.shape)

writer = SummaryWriter("logs")

writer.add_image("train", img_array, 2, dataformats='HWC')        #注意参数里的类型#
for i in range(100):
    writer.add_scalar("y=2*x", 2*i, i)

writer.close()
