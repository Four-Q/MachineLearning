import torchvision
import cifar10_model
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# 相关训练参数
EPOCH = 10
learning_rate = 1e-2
batch_size = 64

# 加载数据集
train_dataset = torchvision.datasets.CIFAR10('./Pytorch/data/CIFAR10', train=True,  \
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10('./Pytorch/data/CIFAR10', train=False,   \
                                            transform=torchvision.transforms.ToTensor(), download=True)

#
train_dataloader = DataLoader(train_dataset, batch_size)
test_dataloader = DataLoader(test_dataset, batch_size)



# 查看数据集大小
print(f'train_dataset size: {len(train_dataset)}')
print(f'test_dataset size: {len(test_dataset)}')

# 创建模型
model = cifar10_model.Cifar10Model()

# 选择损失函数
# 分类问题：选择交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 选择优化器
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 使用tensorboard进行可视化
writer = SummaryWriter(log_dir='./Pytorch/logs/cifar10_logs')

# 开始训练
# 对于每一epoch的训练
for epoch in range(EPOCH):
    print(f'---------epoch{epoch}-----------')
    epoch_loss = 0
    # 对于每一batch的训练集训练
    model.train()       # 注意这一行的作用

    train_batch = 0
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        optim.zero_grad()   # 梯度归零
        loss = loss_func(output, targets)
        epoch_loss += loss
        loss.backward()  # 反向传播计算梯度
        optim.step()     # 更新模型参数
        train_batch += 1
        if train_batch % 100 == 0:
            print(f'    train_batch{train_batch} loss:  {loss.item()}')
            # tensorboard可视化loss
            writer.add_scalar(f'train_epoch{epoch}', loss, global_step=train_batch)

    # 可视化每一epoch下的loss
    writer.add_scalar('epoch_loss', epoch_loss, epoch)

    # 测试集测试
    with torch.no_grad():
        model.eval()
        total_accurates = 0
        total_test_loss = 0
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_func(output, targets)
            accurates = (output.argmax(1) == targets).sum()
            total_test_loss += loss
            total_accurates += accurates

        accuracy = total_accurates/len(test_dataset)
        # 可视化训练集的loss
        writer.add_scalar('epoch_test_loss', total_test_loss, epoch) 
        writer.add_scalar('epoch_test_accuracy', accuracy, epoch)
        print(f'epoch{epoch} test loss: {total_test_loss}')
        print(f'epoch{epoch} test accuracy: {accuracy*100: .2f}%')

    # 保存模型数据
    torch.save(model, f'./Pytorch/model_data/CIFAR10_EPOCH{epoch}.pth')
    break
writer.close()

