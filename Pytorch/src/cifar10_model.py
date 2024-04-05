import torch
from torch import nn

# 模型搭建
class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64), 
            nn.Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output
    


# 模型测试

if __name__ == '__main__':
    input = torch.ones((64, 3, 32, 32))
    model = Cifar10Model()
    output = model(input)
    print(output.shape)