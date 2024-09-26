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

from model import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer
from data import *
from utils import *
from global_context import set_training_step
# from transformers import ViTForImageClassification

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(
            size: int = 'b',
            num_classes: int = 10,
            pretrained: bool = False,
            att_scheme: str = 'gqa',
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
    args = dict(num_classes=num_classes, pretrained=pretrained, att_scheme=att_scheme, window_size=window_size, num_kv_heads=num_kv_heads, in_chans=in_chans)
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
        # model = ViTForImageClassification.from_pretrained('./ViT-base-patch16-224/pytorch_model.bin')

        return model
    else:
        raise ValueError(f'Expected one of s/b/l/c for size - got {size}')
    
def train_step(model, dataloader, criterion, optimizer, device):
    '''Train for one epoch'''

    model.train()

    train_loss = 0.
    train_acc = 0.

    for step, (X, y) in tqdm(enumerate(dataloader), desc="Training", leave=False):
        set_training_step(step)
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred = torch.argmax(logits.detach(), dim=1)
        train_acc += ((y_pred == y).sum().item() / len(y))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the directory where new experiment runs will be tracked')
    parser.add_argument('--save_model', type=bool, default=False, help='Save the model at the end of the training run.')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to .pth file to load in for the training run.')
    args = parser.parse_args()

    config = load_config(args.config)
    
    OUT_DIR_ROOT = args.out_dir
    exp_name = os.path.basename(args.config)
    exp_name = os.path.splitext(exp_name)[0]
    out_dir = os.path.join(OUT_DIR_ROOT,
                           exp_name)
    os.makedirs(os.path.join(out_dir),
                exist_ok=True)
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
    batch_size = 32
    num_workers = 2

    train_ds, test_ds = get_dataset_func(dataset_name)(root=f'./{dataset_name}')
    train_dl = get_dataloader(train_ds, batch_size, True, num_workers)
    test_dl = get_dataloader(test_ds, batch_size, False, num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model, optimizer, loss function setup
    model = get_model(
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
            
            curr_state_dict = model.state_dict()
            new_state_dict = {k: v for k, v in checkpoint.items() if k in curr_state_dict and 'head' not in k}
            curr_state_dict.update(new_state_dict)
            model.load_state_dict(curr_state_dict)
            print(f"Loaded in checkpoint from {args.pretrained_ckpt}!")

    model.to(device)

    learning_rate = 1e-5
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Loaded in model with {count_parameters(model)} parameters...")
    print(f"Using device {device}...")

    # Start the training run and log
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
                    torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))

            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc])
            print(f"{epoch+1=} | {train_acc=} | {test_acc=}")

    # Save this model at the end of run (commented out for)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(out_dir, 'final.pth'))