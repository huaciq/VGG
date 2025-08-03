import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from VGG import VGG16
import torch.optim as optim

# 确定使用硬件，定义超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device :{device}")

LEARNING_RATE = 0.01
BATCH_SIZE = 24
EPOCHS = 50

# 对图片进行一系列处理
transforms_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transforms_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# 加载数据并且分批
trainset = torchvision.datasets.CIFAR10(root=r'F:\python_code\vgg\CIFAR10',train=True,download=True,transform=transforms_train)
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root=r'F:\python_code\vgg\CIFAR10',train=False,download=True,transform=transforms_test)
testloader = DataLoader(testset,batch_size=100,shuffle=False,num_workers=2)


net = VGG16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.parameters(),lr=LEARNING_RATE,momentum=0.9,weigth_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)


def train(epoch):
    # 先给网络设置成训练模式
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs,targets) in enumerate(trainloader):
        inputs,targets = inputs.to(device),targets.to(device)
        # --- 核心训练五步法 ---
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        # 10. 累积统计数据
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 11. 定期打印训练日志
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%')

    return  train_loss/len(trainloader)
