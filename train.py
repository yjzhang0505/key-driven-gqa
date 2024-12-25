import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import argparse
import logging
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import timm
from vitb_my_gqa import VisionTransformer
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
TRAIN_TFMS = transforms.Compose([
    transforms.RandAugment(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
TEST_TFMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 数据集加载
root = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/cifar100'
train_dataset = datasets.CIFAR100(root, train=True, download=True, transform=TRAIN_TFMS)
test_dataset = datasets.CIFAR100(root, train=False, download=True, transform=TEST_TFMS)

# 代理数据集函数
def get_proxy_dataset(dataset, proxy_ratio=0.1):
    dataset_size = len(dataset)
    proxy_size = int(proxy_ratio * dataset_size)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    proxy_indices = indices[:proxy_size]
    return Subset(dataset, proxy_indices)

proxy_ratio = 0.1
proxy_train_dataset = get_proxy_dataset(train_dataset, proxy_ratio=proxy_ratio)
print(f"Using proxy dataset with ratio {proxy_ratio}")

train_loader = DataLoader(proxy_train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义模型
parser = argparse.ArgumentParser(description='put in filepath.')
parser.add_argument('--file_path', type=str, help='Path to the group txt')
args = parser.parse_args()

model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=100,  # CIFAR-100 数据集
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.,
    qkv_bias=True,
    norm_layer=nn.LayerNorm,
    file_path=args.file_path,
)

# 加载检查点
pth_path = "/data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth"
checkpoint = torch.load(pth_path)
model.load_state_dict(checkpoint, strict=False)
model.load_pretrained_weights(checkpoint)
print(f"Loaded pretrained weights from {pth_path}!")

# 将模型移到设备
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 学习率调度器：Cosine Annealing + WarmUp
def get_scheduler(optimizer, num_epochs, warmup_epochs=5):
    # Cosine Annealing学习率调度器
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor(epoch - warmup_epochs) * 3.14159 / (num_epochs - warmup_epochs)))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

# 获取学习率调度器
scheduler = get_scheduler(optimizer, num_epochs=15)

# 创建 TensorBoard 的日志目录
log_dir = 'runs/cifar100_training'  # 可以根据需要调整日志保存路径
writer = SummaryWriter(log_dir)

# 训练函数
def train(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.4f}"})

    # 记录训练损失和精度到 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    return total_loss / total_samples, correct / total_samples

# 测试函数
def test(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)

            avg_loss = total_loss / total_samples
            accuracy = correct / total_samples
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.4f}"})

    # 记录测试损失和精度到 TensorBoard
    writer.add_scalar('Loss/test', avg_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    return total_loss / total_samples, correct / total_samples


# 训练主循环
num_epochs = 15
logging.basicConfig(
    filename=os.path.join(args.file_path, "Result.txt"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
    test_loss, test_acc = test(model, test_loader, criterion, device, epoch)
    scheduler.step()

    logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

logging.info("Training complete. Model saved.")

# 关闭 TensorBoard writer
writer.close()
