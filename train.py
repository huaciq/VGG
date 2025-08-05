import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from VGG import VGG16
import torch.optim as optim
import matplotlib.pyplot as plt

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
trainset = torchvision.datasets.CIFAR10(root=r'F:\python_code\vgg\CIFAR10',train=True,download=False,transform=transforms_train)
trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

testset = torchvision.datasets.CIFAR10(root=r'F:\python_code\vgg\CIFAR10',train=False,download=False,transform=transforms_test)
testloader = DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)


net = VGG16().to(device)
criterion = nn.CrossEntropyLoss()
optimizer =optim.SGD(net.parameters(),lr=LEARNING_RATE,momentum=0.9,weight_decay=5e-4)
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

    return train_loss/len(trainloader)

# 5. 测试函数
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(testloader):
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = net(inputs)
            # 7. 计算损失（仅用于监控，不用于反向传播）
            loss = criterion(outputs,targets)
            # 8. 累积统计数据（损失和准确率）
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct = predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f'---> Test on Epoch {epoch} | Loss: {test_loss/len(testloader):.3f} | Acc: {acc:.3f}%')


    # 10. 返回该 epoch 的平均测试损失和总准确率
    return test_loss/len(testloader), acc


# 6. 主循环和可视化
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(EPOCHS):
    train_loss = train(epoch)
    test_loss, test_acc = test(epoch)

    scheduler.step()    # 更新学习率

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1,2,2)
plt.plot(test_accuracies, label='Testing Accuracy')
plt.title("Accuracy vs. Epochs")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('VGG16_cifar10_training.png')
plt.show()

# 保存模型

torch.save(net.state_dict(),'vgg16_cifar10.pth')
print("Finished Training and saved model.")

