from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("flower_photos/roses/12240303_80d87f77a3_n.jpg")

#ToTensor的使用
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normoalize 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
#     ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
# （input-0.5)/0.5 = 2*input-1 假设input[0,1],则output[-1,1]
print(img_norm[0][0][0])
writer.add_image("img_norm", img_norm)

# Resize的使用
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(128)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
