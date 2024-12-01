import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision
import io
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

from vitb_gqa import VisionTransformer
# from vit_base_patch16_224 import VisionTransformer

IMAGE_SIZE = 224
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

class CIFAR100ParquetDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        # 加载数据
        self.data = pd.read_parquet(parquet_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从 img 列提取字节流
        img_dict = self.data.iloc[idx]["img"]  # 获取字典
        img_bytes = img_dict["bytes"]  # 提取字典中的字节流
        img = Image.open(io.BytesIO(img_bytes))  # 使用 PIL 打开字节流

        # 提取 fine_label 作为标签
        label = self.data.iloc[idx]["fine_label"]

        # 应用变换
        if self.transform:
            img = self.transform(img)

        return img, label



from torch.utils.data import DataLoader
from torchvision import transforms

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),  # 调整到 ViT 模型需要的大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

root = '/data/yjzhang/desktop/try/not_share/key-driven-gqa/cifar100'

train_dataset = torchvision.datasets.CIFAR100(
    root, train=True, download=True, transform=TRAIN_TFMS
)

test_dataset = torchvision.datasets.CIFAR100(
    root, train=False, download=True, transform=TEST_TFMS
)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


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
)

# 检查点路径
pth_path = "/data/yjzhang/desktop/try/ckpt/cifar100/4/model.pth"  # 替换为你的检查点文件路径

 
# 加载检查点
checkpoint = torch.load(pth_path)

model.load_state_dict(checkpoint, strict=False)

model.load_pretrained_weights(checkpoint)
print(f"Loaded pretrained weights from {pth_path}!")



# 将模型移到设备
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)

        # 更新进度条
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.4f}"})

    return total_loss / total_samples, correct / total_samples

# 测试函数
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    pbar = tqdm(loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 记录损失和准确率
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)

            # 更新进度条
            avg_loss = total_loss / total_samples
            accuracy = correct / total_samples
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{accuracy:.4f}"})

    return total_loss / total_samples, correct / total_samples

# 开始训练
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    scheduler.step()

    # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# 保存训练后的模型
# torch.save(model.state_dict(), "vit_cifar100_finetuned.pth")
print("Training complete. Model saved as vit_cifar100_finetuned.pth")
