"""
训练 MNIST 手写数字分类模型
使用 torchvision 下载 MNIST 数据集，训练 CNN 模型并保存权重
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='训练 MNIST 手写数字分类模型')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数 (默认: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小 (默认: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (默认: 0.001)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='设备类型 (默认: cpu)')
    parser.add_argument('--save_path', type=str, default='weights/mnist_cnn.pt',
                       help='模型保存路径 (默认: weights/mnist_cnn.pt)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 数据预处理：与 MNIST 标准一致
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
    ])
    
    # 加载 MNIST 数据集
    print("正在下载/加载 MNIST 数据集...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    model = get_model(num_classes=10, device=device) # type: ignore
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    print(f"\n开始训练 ({args.epochs} epochs)...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 打印结果
        print(f"Epoch [{epoch}/{args.epochs}]")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print("-" * 60)
    
    # 保存模型
    torch.save(model.state_dict(), args.save_path)
    print(f"\n模型已保存到: {args.save_path}")


if __name__ == '__main__':
    main()

