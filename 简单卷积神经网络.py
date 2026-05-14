import torch.nn as nn

# 一个简单的CNN示例（用于图像分类）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 256) # 假设经过两次池化后特征图大小为8x8
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x))) # 卷积 -> 激活 -> 池化
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8) # 将特征图拍平成一维向量
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x