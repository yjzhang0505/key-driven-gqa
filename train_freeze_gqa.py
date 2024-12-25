
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# import timm

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

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# import timm
import random
from typing import Optional
import shutil
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# 添加早停
class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='checkpoint.pth'):
        self.patience = patience  # 允许的容忍次数
        self.delta = delta  # 需要的最小改善量
        self.path = path  # 检查点保存路径
        self.best_acc = None  # 用于存储最佳准确率
        self.epochs_without_improvement = 0  # 没有改进的周期数

    def __call__(self, test_acc, model):
        # 如果没有记录最佳准确率，初始化
        if self.best_acc is None:
            self.best_acc = test_acc
            self.save_checkpoint(model)
        # 如果当前准确率比最佳准确率高，并且高于 `delta`，则更新最佳准确率
        elif test_acc > self.best_acc + self.delta:
            self.best_acc = test_acc
            self.epochs_without_improvement = 0
            self.save_checkpoint(model)
        else:
            # 如果没有改善，增加未改进的周期数
            self.epochs_without_improvement += 1
            # 如果连续 `patience` 个周期没有改进，执行早停
            if self.epochs_without_improvement >= self.patience:
                print("Early stopping")
                return True
        return False

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


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

import random
test_dataset = torchvision.datasets.CIFAR100(
    root, train=False, download=True, transform=TEST_TFMS
)
def get_proxy_dataset(dataset, proxy_ratio=0.1):
    """
    从原始数据集中随机选择一定比例的样本，创建代理数据集。
    
    :param dataset: 原始数据集
    :param proxy_ratio: 代理数据集占原始数据集的比例
    :return: Subset对象，表示代理数据集
    """
    dataset_size = len(dataset)
    proxy_size = int(proxy_ratio * dataset_size)
    
    # 随机选择代理数据集的样本索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    proxy_indices = indices[:proxy_size]

    # 返回代理数据集
    proxy_dataset = Subset(dataset, proxy_indices)
    return proxy_dataset

set_seed()
proxy_ratio = 0.5
proxy_train_dataset = get_proxy_dataset(train_dataset, proxy_ratio=proxy_ratio)
print(f"Using proxy dataset with ratio {proxy_ratio}")

# 定义数据加载器
train_loader = DataLoader(proxy_train_dataset, batch_size=32, shuffle=True)
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

# model = torch.nn.DataParallel(model, device_ids=[0,1,2])  # 指定 GPU 设备 0, 1

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

early_stopping = EarlyStopping(patience=5, path='early_stopped_model.pth')

# 开始训练
num_epochs = 100
import logging

# 配置 logging
logging.basicConfig(
    filename="/data/yjzhang/desktop/try/not_share/key-driven-gqa/output/dustbin2/gqa/proxy=0.5.txt",  # 输出到的文件
    level=logging.INFO,           # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    scheduler.step()

    logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    # 使用验证损失进行早停判断
    if early_stopping(test_acc, model):
        break  # 提前停止训练    