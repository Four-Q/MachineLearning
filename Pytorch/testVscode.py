import torch
from torchvision import transforms
from PIL import Image

image_path = './pytorch/data/Image/see.jpg'


image = Image.open(image_path)


image.show()
