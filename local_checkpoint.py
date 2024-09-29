import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from torchvision import datasets, transforms
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

def train_vit_from_checkpoint(checkpoint_path, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cuda'):
    """
    加载本地 ViT-base-patch16-224 的检查点，并在其基础上继续训练。
    
    Args:
        checkpoint_path (str): 本地存储的检查点路径 (如 '/path/to/pytorch_model.bin')
        train_loader (DataLoader): 训练集的数据加载器
        val_loader (DataLoader): 验证集的数据加载器
        num_epochs (int): 训练的轮数
        lr (float): 学习率
        device (str): 设备，'cuda' 或 'cpu'
        
    Returns:
        model (torch.nn.Module): 经过训练后的模型
    """
    # 1. 创建 ViT-base-patch16-224 模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
    
    # 2. 加载本地检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # 加载权重到模型中
    
    # 3. 将模型移到设备（CPU 或 GPU）
    model = model.to(device)
    
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 假设任务是分类任务
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 使用 tqdm 显示训练进度条
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # 计算训练精度
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                # 每个 batch 更新进度条显示的损失
                tepoch.set_postfix(loss=running_loss / (total_train), accuracy=100. * correct_train / total_train)
        
        # 计算每个 epoch 的平均损失和精度
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
        # 验证模型
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    
    return model

def validate_model(model, val_loader, criterion, device):
    """
    验证模型性能。
    
    Args:
        model (torch.nn.Module): 要验证的模型
        val_loader (DataLoader): 验证集的数据加载器
        criterion (Loss): 损失函数
        device (str): 设备，'cuda' 或 'cpu'
    
    Returns:
        val_loss (float): 验证集的平均损失
        val_acc (float): 验证集的准确率
    """
    model.eval()  # 设置模型为评估模式
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Validating", unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 计算验证精度
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                # 每个 batch 更新进度条显示的验证损失
                tepoch.set_postfix(loss=val_loss / total_val, accuracy=100. * correct_val / total_val)
    
    # 计算验证集的平均损失和精度
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct_val / total_val
    return val_loss, val_acc

if __name__ == "__main__":
    # 参数设置
    # checkpoint_path = '/data/yjzhang/desktop/checkpoint/pytorch_model.bin'
    checkpoint_path = './checkpoint/finetuned_vit_base_patch16_224.pth'
    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 使用示例数据集 (CIFAR-10)
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 加载检查点并开始训练
    model = train_vit_from_checkpoint(
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device
    )

    # 训练后的模型保存在本地
    torch.save(model.state_dict(), "finetuned_vit_base_patch16_224.pth")
    print("训练完成，模型已保存为 finetuned_vit_base_patch16_224.pth")
