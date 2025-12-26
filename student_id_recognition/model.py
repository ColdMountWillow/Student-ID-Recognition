"""
简单的 CNN 模型用于手写数字识别（0-9）
输入：1x28x28 的灰度图像
输出：10 类 logits
"""
import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """简单 CNN 模型：3 层 Conv-ReLU-Pool + 全连接层"""
    
    def __init__(self, num_classes=10):
        super(DigitCNN, self).__init__()
        
        # 第一层卷积：1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        
        # 第二层卷积：32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        
        # 第三层卷积：64 -> 128 channels，增强表达能力
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7 -> 3x3 (向下取整)
        
        # 全连接层
        # 经过 3 次池化：28 -> 14 -> 7 -> 3，所以是 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 128*3*3]
        
        # 全连接层
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10, device='cpu'):
    """
    返回模型实例
    
    Args:
        num_classes: 分类数量，默认 10（0-9）
        device: 设备，'cpu' 或 'cuda'
    
    Returns:
        DigitCNN 模型实例
    """
    model = DigitCNN(num_classes=num_classes)
    model = model.to(device)
    return model

