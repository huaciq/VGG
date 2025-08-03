import torch
import torch.nn as nn
from torchvision.models import vgg16

# 这是论文 Table 1, "D" 列的配置
# 数字代表卷积层的输出通道数，'M' 代表最大池化层
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG16,self).__init__()
        self.flatten = nn.Flatten(start_dim=1)

        # 'features' 部分，即卷积层和池化层
        # 池化一次，尺寸一般就减半，计算公式同卷积，但是padding一般为0
        self.features = self._make_layers(vgg16_config)
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )


    # 前向传播
    def forward(self,x):
        x = self.features(x)
        x = self.flatten(x)  # shape: 512*1*1
        x = self.classifier(x)
        return x

    def _make_layers(self, config):
        layers = []
        in_channels = 3  # RGB 图像的输入通道是 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = VGG16().to(device)
    # print(net)
    #
    # print("--- PyTorch Official VGG16 for ImageNet ---")
    # print(vgg16())
    #
    # print("---My Vgg16net---")
    # print(VGG16())

    dummy_input = torch.randn(2,3,32,32).to(device)
    output = net(dummy_input)
    print(f"\nInput shape:{dummy_input.shape}")
    print(f"Output shape:{output.shape}")