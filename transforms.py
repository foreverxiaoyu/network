from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#import cv2

img_path = "flower_photos/roses/24781114_bc83aa811e_n.jpg"
img_path_abs = "/home/dell/learning/data/flower_photos/roses/24781114_bc83aa811e_n.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()
#cv_img = cv2.imread(img_path)

print(tensor_img)
