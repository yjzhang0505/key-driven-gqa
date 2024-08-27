import os

import yaml
import torch
import torch.nn as nn

def load_config(config_path: str):
    assert os.path.exists(config_path), "The provided path is invalid"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def assign_check(left: torch.Tensor, right: torch.Tensor):
    '''Utility for checking and creating parameters to be used in loading a pretrained checkpoint'''
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert value {s} to bool")