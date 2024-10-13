from typing import Optional
import shutil
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Subset

from model import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer
from data import *
from utils import *
from global_context import set_training_step

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(
            exp_num : int,
            size: int = 's',
            num_classes: int = 10,
            pretrained: bool = False,
            att_scheme: str = 'mhsa',
            window_size: int = 10,
            num_kv_heads: int = 2,
            in_chans: int = 3,
            embed_dim: Optional[int] = None,
            num_layers: Optional[int] = None,
            num_heads: Optional[int] = None
            ):
    '''
    Model factory function for loading in models according to pretrained checkpoints or a custom model to be trained from scratch    
    '''
    args = dict(num_classes=num_classes, pretrained=pretrained, att_scheme=att_scheme, window_size=window_size, num_kv_heads=num_kv_heads, exp_num=exp_num, in_chans=in_chans)
    if size == 's':
        print(f"Loaded in small ViT with args {args}")
        return vit_small_patch16_224(**args)
    elif size == 'b':
        print(f"Loaded in base ViT with args {args}")
        return vit_base_patch16_224(**args)
    elif size == 'l':
        print(f"Loaded in large ViT with args {args}")
        return vit_large_patch16_224(**args)
    
    # Add logic for loading in a custom model which isn't pretrained
    elif size == 'c':
        assert pretrained == False, "Cannot load in a pretrained ckpt for a custom model"
        assert all([x is not None for x in [embed_dim, num_layers, num_heads]]), "Provide all the optional arguments when creating a custom model"
        model = VisionTransformer(
            exp_num = 0,
            img_size=224,
            patch_size=16,
            in_chans=in_chans,
            num_classes=num_classes,
            num_kv_heads=num_kv_heads,
            window_size=window_size,
            att_scheme=att_scheme,
            embed_dim=embed_dim,
            depth=num_layers,
            num_heads=num_heads
        )
        return model
    else:
        raise ValueError(f'Expected one of s/b/l/c for size - got {size}')
    
def train_step(model, dataloader, criterion, optimizer, device):
    '''训练一个epoch'''

    model.train()

    train_loss = 0.
    train_acc = 0.

    # 使用tqdm进度条，并在每次更新时显示当前批次的精度和损失
    progress_bar = tqdm(enumerate(dataloader), desc="训练中", leave=False, total=len(dataloader))

    for step, (X, y) in progress_bar:
        set_training_step(step)  # 假设您已经定义了set_training_step函数
        X, y = X.to(device), y.to(device)

        # 前向传播
        logits = model(X)
        loss = criterion(logits, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        optimizer.step()

        # 计算当前批次的准确率
        y_pred = torch.argmax(logits.detach(), dim=1)
        batch_acc = (y_pred == y).sum().item() / len(y)
        train_acc += batch_acc

        # 更新进度条描述信息，显示当前批次的准确率和平均损失
        progress_bar.set_postfix({
            "loss": train_loss / (step + 1),  # 平均损失
            "accuracy": train_acc / (step + 1)  # 平均精度
        })

    # 计算平均损失和准确率
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

@torch.inference_mode()
def eval_step(model, dataloader, criterion, device):
    
    model.eval()

    eval_loss = 0.
    eval_acc = 0.

    for (X, y) in tqdm(dataloader, desc="Evaluating", leave=False):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        eval_loss += loss.item()

        y_pred = torch.argmax(logits.detach(), dim=1)
        eval_acc += ((y_pred == y).sum().item() / len(y))

    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / len(dataloader)
    return eval_loss, eval_acc

def merge_qkv_weights(model_state_dict):
    """
    将 q, k, v 的权重和偏置合并为 qkv。
    """
    # 创建一个新的 state_dict 的副本
    merged_state_dict = model_state_dict.copy()

    # 遍历所有block，合并q, k, v
    for i in range(12):  # 假设 ViT-B有12个Block
        q_weight = model_state_dict[f'blocks.{i}.attn.q.weight']
        k_weight = model_state_dict[f'blocks.{i}.attn.k.weight']
        v_weight = model_state_dict[f'blocks.{i}.attn.v.weight']

        q_bias = model_state_dict[f'blocks.{i}.attn.q.bias']
        k_bias = model_state_dict[f'blocks.{i}.attn.k.bias']
        v_bias = model_state_dict[f'blocks.{i}.attn.v.bias']

        # 合并权重
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

        # 将合并后的qkv替换到新的state_dict
        merged_state_dict[f'blocks.{i}.attn.qkv.weight'] = qkv_weight
        merged_state_dict[f'blocks.{i}.attn.qkv.bias'] = qkv_bias

        # 删除原有的 q, k, v
        del merged_state_dict[f'blocks.{i}.attn.q.weight']
        del merged_state_dict[f'blocks.{i}.attn.k.weight']
        del merged_state_dict[f'blocks.{i}.attn.v.weight']

        del merged_state_dict[f'blocks.{i}.attn.q.bias']
        del merged_state_dict[f'blocks.{i}.attn.k.bias']
        del merged_state_dict[f'blocks.{i}.attn.v.bias']

    return merged_state_dict


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train Vision Transformer')
#     parser.add_argument('--config', type=str, required=True, help='Path to the config file')
#     parser.add_argument('--out_dir', type=str, required=True, help='Path to the directory where new experiment runs will be tracked')
#     parser.add_argument('--save_model', type=bool, default=False, help='Save the model at the end of the training run.')
#     parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to .pth file to load in for the training run.')
#     parser.add_argument('--exp_num', type=str, default=0, help='Num of .pth file to load in for the training run.')
#     args = parser.parse_args()

#     config = load_config(args.config)
    
#     # OUT_DIR_ROOT = args.out_dir
#     # exp_name = os.path.basename(args.config)
#     # exp_name = os.path.splitext(exp_name)[0]
#     # out_dir = os.path.join(OUT_DIR_ROOT,
#     #                        exp_name)
#     out_dir = args.out_dir
#     os.makedirs(os.path.join(out_dir),
#                 exist_ok=True)
#     print(f"Results for {os.path.basename(args.config)} being saved to {out_dir}...")

#     if args.save_model:
#         print("Going to save model at end of run...")

#     # Copy over the yaml file
#     shutil.copyfile(
#         args.config,
#         os.path.join(out_dir, 'config.yaml')
#     )
    
#     # Set seed for reproducibility
#     set_seed()
    
#     # Data setup
#     dataset_name = config['dataset']
#     batch_size = 32
#     num_workers = 2

#     train_ds, test_ds = get_dataset_func(dataset_name)(root=f'./{dataset_name}')
#     train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
#     test_dl = get_dataloader(test_ds, batch_size, False, num_workers)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Model, optimizer, loss function setup
#     model = get_model(
#         exp_num = args.exp_num,
#         size=config['size'], 
#         num_classes=config['num_classes'], 
#         pretrained=config['pretrained'], 
#         att_scheme=config['att_scheme'],
#         window_size=config['window_size'],
#         num_kv_heads=config['num_kv_heads'],
#         in_chans=config['in_chans'],
#         embed_dim=config.get('embed_dim', None),
#         num_layers=config.get('num_layers', None),
#         num_heads=config.get('num_heads', None)
#     )

#     # Load in pretrained weight if any
#     if args.pretrained_ckpt:
#         if os.path.exists(args.pretrained_ckpt):
#             checkpoint = torch.load(args.pretrained_ckpt)
#             state_dict = checkpoint
#             del checkpoint['head.weight']
#             del checkpoint['head.bias']

            
#             # curr_state_dict = model.state_dict()
            
#             # new_state_dict = {k: v for k, v in checkpoint.items() if k in state_dict and 'head' not in k}
#             # state_dict.update(new_state_dict)
#             # model.load_state_dict(curr_state_dict)
#             model.load_pretrained_weights(state_dict)
#             # model.load_state_dict(state_dict, strict=False)
#             model.head.weight.data = torch.randn_like(model.head.weight)
#             model.head.bias.data = torch.randn_like(model.head.bias)            
#             print(f"Loaded in checkpoint from {args.pretrained_ckpt}!")

#     model.to(device)

#     learning_rate = 1e-5
#     epochs = 500
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#     print(f"Loaded in model with {count_parameters(model)} parameters...")
#     print(f"Using device {device}...")

#     # Start the training run and log
#     # best_loss = float('inf')
#     # with open(os.path.join(out_dir, 'run.csv'), 'w') as f:
#     #     writer = csv.writer(f)
#     #     writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

#     #     for epoch in tqdm(range(epochs), desc="Epochs"):

#     #         train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
#     #         test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
#     #         if test_loss < best_loss:
#     #             best_loss = test_loss
#     #             if args.save_model:
#     #                 state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#     #                 merged_state_dict = merge_qkv_weights(state_dict)
#     #                 torch.save(merged_state_dict, os.path.join(out_dir, 'best.pth'))
#     #                 # torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))

#     #         writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
#     #         print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

#     # # Save this model at the end of run (commented out for)
#     # if args.save_model:
#     #     state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#     #     merged_state_dict = merge_qkv_weights(state_dict)
#     #     torch.save(merged_state_dict, os.path.join(out_dir, 'final.pth'))
#     #     # torch.save(model.state_dict(), os.path.join(out_dir, 'final.pth'))

#     best_loss = float('inf')

#     with open(os.path.join(out_dir, 'run.csv'), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

#         for epoch in tqdm(range(epochs), desc="Epochs"):

#             train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
#             test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
#             if test_loss < best_loss:
#                 best_loss = test_loss
#                 if args.save_model:
#                     state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#                     merged_state_dict = merge_qkv_weights(state_dict)
#                     torch.save(merged_state_dict, os.path.join(out_dir, 'best.pth'))

#             # 记录每个epoch的损失和准确率
#             writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
#             print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

#             # 检查第3个epoch的训练准确率
#             if epoch + 1 == 3 and test_acc < 0.4:
#                 print(f"第{epoch + 1}个epoch的训练准确率 {train_acc} 小于 0.786，提前停止训练。")
#                 break  # 提前停止训练

#     # 在训练结束时保存模型
#     if args.save_model:
#         state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#         merged_state_dict = merge_qkv_weights(state_dict)
#         torch.save(merged_state_dict, os.path.join(out_dir, 'final.pth'))


def get_proxy_dataset(dataset, proxy_ratio=0.1):
    """
    从原始数据集中随机选择一定比例的样本，创建代理数据集。
    
    :param dataset: 原始数据集
    :param proxy_ratio: 代理数据集占原始数据集的比例
    :return: Subset对象，表示代理数据集
    """
    set_seed()
    dataset_size = len(dataset)
    proxy_size = int(proxy_ratio * dataset_size)
    
    # 随机选择代理数据集的样本索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    proxy_indices = indices[:proxy_size]

    # 返回代理数据集
    proxy_dataset = Subset(dataset, proxy_indices)
    return proxy_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the directory where new experiment runs will be tracked')
    parser.add_argument('--save_model', type=bool, default=False, help='Save the model at the end of the training run.')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to .pth file to load in for the training run.')
    parser.add_argument('--exp_num', type=str, default=0, help='Num of .pth file to load in for the training run.')
    parser.add_argument('--proxy_ratio', type=float, default=1.0, help='Ratio of the proxy dataset to use (1.0 for full dataset)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir), exist_ok=True)
    print(f"Results for {os.path.basename(args.config)} being saved to {out_dir}...")

    if args.save_model:
        print("Going to save model at end of run...")

    # Copy over the yaml file
    shutil.copyfile(
        args.config,
        os.path.join(out_dir, 'config.yaml')
    )
    
    # Set seed for reproducibility
    set_seed()
    
    # Data setup
    dataset_name = config['dataset']
    batch_size = 64
    num_workers = 2

    # 获取数据集
    train_ds, test_ds = get_dataset_func(dataset_name)(root=f'./{dataset_name}')
    
    # 应用 proxy_ratio 参数，缩小数据集
    if args.proxy_ratio < 1.0:
        print(f"Using proxy dataset with ratio {args.proxy_ratio}")
        train_ds = get_proxy_dataset(train_ds, proxy_ratio=args.proxy_ratio)
    
    # 创建数据加载器
    train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
    test_dl = get_dataloader(test_ds, batch_size, False, num_workers)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model, optimizer, loss function setup
    model = get_model(
        exp_num = args.exp_num,
        size=config['size'], 
        num_classes=config['num_classes'], 
        pretrained=config['pretrained'], 
        att_scheme=config['att_scheme'],
        window_size=config['window_size'],
        num_kv_heads=config['num_kv_heads'],
        in_chans=config['in_chans'],
        embed_dim=config.get('embed_dim', None),
        num_layers=config.get('num_layers', None),
        num_heads=config.get('num_heads', None)
    )

    # Load in pretrained weight if any
    if args.pretrained_ckpt:
        if os.path.exists(args.pretrained_ckpt):
            checkpoint = torch.load(args.pretrained_ckpt)
            state_dict = checkpoint
            del checkpoint['head.weight']
            del checkpoint['head.bias']

            model.load_pretrained_weights(state_dict)
            model.head.weight.data = torch.randn_like(model.head.weight)
            model.head.bias.data = torch.randn_like(model.head.bias)            
            print(f"Loaded in checkpoint from {args.pretrained_ckpt}!")

    model.to(device)

    learning_rate = 1e-5
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Loaded in model with {count_parameters(model)} parameters...")
    print(f"Using device {device}...")

    best_loss = float('inf')

    with open(os.path.join(out_dir, 'run.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

        for epoch in tqdm(range(epochs), desc="Epochs"):
            train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
            test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
            if test_loss < best_loss:
                best_loss = test_loss
                if args.save_model:
                    state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
                    merged_state_dict = merge_qkv_weights(state_dict)
                    torch.save(merged_state_dict, os.path.join(out_dir, 'best.pth'))

            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
            print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

            if epoch + 1 == 3 and test_acc < 0.4:
                print(f"第{epoch + 1}个epoch的训练准确率 {train_acc} 小于 0.786，提前停止训练。")
                break  # 提前停止训练

    if args.save_model:
        state_dict = model.state_dict()
        merged_state_dict = merge_qkv_weights(state_dict)
        torch.save(merged_state_dict, os.path.join(out_dir, 'final.pth'))
# from typing import Optional
# import shutil
# import argparse
# import csv
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from model import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer
# from data import *
# from utils import *
# from global_context import set_training_step

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def get_model(
#             exp_num : int,
#             size: int = 's',
#             num_classes: int = 10,
#             pretrained: bool = False,
#             att_scheme: str = 'mhsa',
#             window_size: int = 10,
#             num_kv_heads: int = 2,
#             in_chans: int = 3,
#             embed_dim: Optional[int] = None,
#             num_layers: Optional[int] = None,
#             num_heads: Optional[int] = None
#             ):
#     '''
#     Model factory function for loading in models according to pretrained checkpoints or a custom model to be trained from scratch    
#     '''
#     args = dict(num_classes=num_classes, pretrained=pretrained, att_scheme=att_scheme, window_size=window_size, num_kv_heads=num_kv_heads, exp_num=exp_num, in_chans=in_chans)
#     if size == 's':
#         print(f"Loaded in small ViT with args {args}")
#         return vit_small_patch16_224(**args)
#     elif size == 'b':
#         print(f"Loaded in base ViT with args {args}")
#         return vit_base_patch16_224(**args)
#     elif size == 'l':
#         print(f"Loaded in large ViT with args {args}")
#         return vit_large_patch16_224(**args)
    
#     # Add logic for loading in a custom model which isn't pretrained
#     elif size == 'c':
#         assert pretrained == False, "Cannot load in a pretrained ckpt for a custom model"
#         assert all([x is not None for x in [embed_dim, num_layers, num_heads]]), "Provide all the optional arguments when creating a custom model"
#         model = VisionTransformer(
#             exp_num = 0,
#             img_size=224,
#             patch_size=16,
#             in_chans=in_chans,
#             num_classes=num_classes,
#             num_kv_heads=num_kv_heads,
#             window_size=window_size,
#             att_scheme=att_scheme,
#             embed_dim=embed_dim,
#             depth=num_layers,
#             num_heads=num_heads
#         )
#         return model
#     else:
#         raise ValueError(f'Expected one of s/b/l/c for size - got {size}')
    
# def train_step(model, dataloader, criterion, optimizer, device):
#     '''训练一个epoch'''

#     model.train()

#     train_loss = 0.
#     train_acc = 0.

#     # 使用tqdm进度条，并在每次更新时显示当前批次的精度和损失
#     progress_bar = tqdm(enumerate(dataloader), desc="训练中", leave=False, total=len(dataloader))

#     for step, (X, y) in progress_bar:
#         set_training_step(step)  # 假设您已经定义了set_training_step函数
#         X, y = X.to(device), y.to(device)

#         # 前向传播
#         logits = model(X)
#         loss = criterion(logits, y)

#         train_loss += loss.item()

#         optimizer.zero_grad()

#         # 反向传播
#         loss.backward()

#         optimizer.step()

#         # 计算当前批次的准确率
#         y_pred = torch.argmax(logits.detach(), dim=1)
#         batch_acc = (y_pred == y).sum().item() / len(y)
#         train_acc += batch_acc

#         # 更新进度条描述信息，显示当前批次的准确率和平均损失
#         progress_bar.set_postfix({
#             "loss": train_loss / (step + 1),  # 平均损失
#             "accuracy": train_acc / (step + 1)  # 平均精度
#         })

#     # 计算平均损失和准确率
#     train_loss = train_loss / len(dataloader)
#     train_acc = train_acc / len(dataloader)

#     return train_loss, train_acc

# @torch.inference_mode()
# def eval_step(model, dataloader, criterion, device):
    
#     model.eval()

#     eval_loss = 0.
#     eval_acc = 0.

#     for (X, y) in tqdm(dataloader, desc="Evaluating", leave=False):
#         X, y = X.to(device), y.to(device)

#         logits = model(X)
#         loss = criterion(logits, y)

#         eval_loss += loss.item()

#         y_pred = torch.argmax(logits.detach(), dim=1)
#         eval_acc += ((y_pred == y).sum().item() / len(y))

#     eval_loss = eval_loss / len(dataloader)
#     eval_acc = eval_acc / len(dataloader)
#     return eval_loss, eval_acc

# def merge_qkv_weights(model_state_dict):
#     """
#     将 q, k, v 的权重和偏置合并为 qkv。
#     """
#     # 创建一个新的 state_dict 的副本
#     merged_state_dict = model_state_dict.copy()

#     # 遍历所有block，合并q, k, v
#     for i in range(12):  # 假设 ViT-B有12个Block
#         q_weight = model_state_dict[f'blocks.{i}.attn.q.weight']
#         k_weight = model_state_dict[f'blocks.{i}.attn.k.weight']
#         v_weight = model_state_dict[f'blocks.{i}.attn.v.weight']

#         q_bias = model_state_dict[f'blocks.{i}.attn.q.bias']
#         k_bias = model_state_dict[f'blocks.{i}.attn.k.bias']
#         v_bias = model_state_dict[f'blocks.{i}.attn.v.bias']

#         # 合并权重
#         qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
#         qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

#         # 将合并后的qkv替换到新的state_dict
#         merged_state_dict[f'blocks.{i}.attn.qkv.weight'] = qkv_weight
#         merged_state_dict[f'blocks.{i}.attn.qkv.bias'] = qkv_bias

#         # 删除原有的 q, k, v
#         del merged_state_dict[f'blocks.{i}.attn.q.weight']
#         del merged_state_dict[f'blocks.{i}.attn.k.weight']
#         del merged_state_dict[f'blocks.{i}.attn.v.weight']

#         del merged_state_dict[f'blocks.{i}.attn.q.bias']
#         del merged_state_dict[f'blocks.{i}.attn.k.bias']
#         del merged_state_dict[f'blocks.{i}.attn.v.bias']

#     return merged_state_dict


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Train Vision Transformer')
#     parser.add_argument('--config', type=str, required=True, help='Path to the config file')
#     parser.add_argument('--out_dir', type=str, required=True, help='Path to the directory where new experiment runs will be tracked')
#     parser.add_argument('--save_model', type=bool, default=False, help='Save the model at the end of the training run.')
#     parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to .pth file to load in for the training run.')
#     parser.add_argument('--exp_num', type=str, default=0, help='Num of .pth file to load in for the training run.')
#     args = parser.parse_args()

#     config = load_config(args.config)
    
#     # OUT_DIR_ROOT = args.out_dir
#     # exp_name = os.path.basename(args.config)
#     # exp_name = os.path.splitext(exp_name)[0]
#     # out_dir = os.path.join(OUT_DIR_ROOT,
#     #                        exp_name)
#     out_dir = args.out_dir
#     os.makedirs(os.path.join(out_dir),
#                 exist_ok=True)
#     print(f"Results for {os.path.basename(args.config)} being saved to {out_dir}...")

#     if args.save_model:
#         print("Going to save model at end of run...")

#     # Copy over the yaml file
#     shutil.copyfile(
#         args.config,
#         os.path.join(out_dir, 'config.yaml')
#     )
    
#     # Set seed for reproducibility
#     set_seed()
    
#     # Data setup
#     dataset_name = config['dataset']
#     batch_size = 32
#     num_workers = 2

#     train_ds, test_ds = get_dataset_func(dataset_name)(root=f'./{dataset_name}')
#     train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
#     test_dl = get_dataloader(test_ds, batch_size, False, num_workers)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Model, optimizer, loss function setup
#     model = get_model(
#         exp_num = args.exp_num,
#         size=config['size'], 
#         num_classes=config['num_classes'], 
#         pretrained=config['pretrained'], 
#         att_scheme=config['att_scheme'],
#         window_size=config['window_size'],
#         num_kv_heads=config['num_kv_heads'],
#         in_chans=config['in_chans'],
#         embed_dim=config.get('embed_dim', None),
#         num_layers=config.get('num_layers', None),
#         num_heads=config.get('num_heads', None)
#     )

#     # Load in pretrained weight if any
#     if args.pretrained_ckpt:
#         if os.path.exists(args.pretrained_ckpt):
#             checkpoint = torch.load(args.pretrained_ckpt)
#             state_dict = checkpoint
#             del checkpoint['head.weight']
#             del checkpoint['head.bias']

            
#             # curr_state_dict = model.state_dict()
            
#             # new_state_dict = {k: v for k, v in checkpoint.items() if k in state_dict and 'head' not in k}
#             # state_dict.update(new_state_dict)
#             # model.load_state_dict(curr_state_dict)
#             model.load_pretrained_weights(state_dict)
#             # model.load_state_dict(state_dict, strict=False)
#             model.head.weight.data = torch.randn_like(model.head.weight)
#             model.head.bias.data = torch.randn_like(model.head.bias)            
#             print(f"Loaded in checkpoint from {args.pretrained_ckpt}!")

#     model.to(device)

#     learning_rate = 1e-5
#     epochs = 5
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#     print(f"Loaded in model with {count_parameters(model)} parameters...")
#     print(f"Using device {device}...")

#     # Start the training run and log
#     # best_loss = float('inf')
#     # with open(os.path.join(out_dir, 'run.csv'), 'w') as f:
#     #     writer = csv.writer(f)
#     #     writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

#     #     for epoch in tqdm(range(epochs), desc="Epochs"):

#     #         train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
#     #         test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
#     #         if test_loss < best_loss:
#     #             best_loss = test_loss
#     #             if args.save_model:
#     #                 state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#     #                 merged_state_dict = merge_qkv_weights(state_dict)
#     #                 torch.save(merged_state_dict, os.path.join(out_dir, 'best.pth'))
#     #                 # torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))

#     #         writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
#     #         print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

#     # # Save this model at the end of run (commented out for)
#     # if args.save_model:
#     #     state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#     #     merged_state_dict = merge_qkv_weights(state_dict)
#     #     torch.save(merged_state_dict, os.path.join(out_dir, 'final.pth'))
#     #     # torch.save(model.state_dict(), os.path.join(out_dir, 'final.pth'))

#     best_loss = float('inf')

#     with open(os.path.join(out_dir, 'run.csv'), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

#         for epoch in tqdm(range(epochs), desc="Epochs"):

#             train_loss, train_acc = train_step(model, train_dl, criterion, optimizer, device)
#             test_loss, test_acc = eval_step(model, test_dl, criterion, device)
            
#             if test_loss < best_loss:
#                 best_loss = test_loss
#                 if args.save_model:
#                     state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#                     merged_state_dict = merge_qkv_weights(state_dict)
#                     torch.save(merged_state_dict, os.path.join(out_dir, 'best.pth'))

#             # 记录每个epoch的损失和准确率
#             writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
#             print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

#             # 检查第3个epoch的训练准确率
#             if epoch + 1 == 3 and train_acc < 0.786:
#                 print(f"第{epoch + 1}个epoch的训练准确率 {train_acc} 小于 0.786，提前停止训练。")
#                 break  # 提前停止训练

#     # 在训练结束时保存模型
#     if args.save_model:
#         state_dict = model.state_dict()  # 确保获取的是模型的state_dict，而不是函数
#         merged_state_dict = merge_qkv_weights(state_dict)
#         torch.save(merged_state_dict, os.path.join(out_dir, 'final.pth'))