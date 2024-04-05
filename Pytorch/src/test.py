import cifar10_model
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 构建模型
# 在gpu上训练的模型在cpu上测试，记得添加map_location= torch.device('cpu')
# map_location 参数允许你在加载模型或张量时指定目标设备，这样可以更灵活地在不同的设备之间转移模型或张量。
model_epoch_9 = torch.load('Pytorch\model_data\CIFAR10_EPOCH29.pth', map_location= torch.device('cpu'))

# 类别：
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 转换器
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
# tensorboard可视化
writer = SummaryWriter(log_dir='./Pytorch/logs/test_logs/')

add_graph = True
accurates = 0

step = 0
for class_ in classes:
    image = Image.open(fr'.\Pytorch\data\CIFAR10\test_image\{class_}.png')
    # 仅使用颜色通道
    image = image.convert('RGB')
    image = trans(image)
    image = torch.reshape(image, (1, 3, 32, 32))
    print(image.shape)
    model_epoch_9.eval()
    if add_graph:
         writer.add_graph(model_epoch_9, input_to_model=image)
         add_graph = False

    with torch.no_grad():
            output = model_epoch_9(image)
    prec_class = classes[output.argmax(1)]
    print(f'prec_class: {prec_class}, actual_class: {class_}')
    if prec_class == class_:
        accurates += 1

    writer.add_image('test', torch.reshape(image, (3, 32, 32)), step)
    step += 1

print(f'accuracy: {accurates/len(classes)}')
