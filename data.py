import os

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, default_collate
from datasets import load_dataset

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

def get_dataset_func(name):
    if name == 'mnist':
        return get_mnist_dataset
    elif name == 'cifar10':
        return get_cifar10_dataset
    elif name == 'cifar100':
        return get_cifar100_dataset
    elif name == 'tiny-imagenet-200':
        return get_tinynet_dataset
    elif name == 'oxfordiitpet':
        return get_oxfordiitpet_dataset
    elif name == 'food101':
        return get_food101_dataset
    elif name == 'cinic10':
        return get_cinic10_dataset
    elif name == 'imagenet-1k':
        return get_imagenet1k_dataset
    else:
        raise ValueError("Received invalid dataset name - please check data.py")
    
def get_dataloader(dataset: Dataset,
                   batch_size: int,
                   is_train: bool,
                   num_workers: int = 1):
    
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=is_train, num_workers=num_workers)
    return loader

def get_mnist_dataset(root: str):

    trainset = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_cifar10_dataset(root: str):

    trainset = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_cifar100_dataset(root: str):

    trainset = torchvision.datasets.CIFAR100(
        root, train=True, download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.CIFAR100(
        root, train=False, download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_food101_dataset(root: str):

    trainset = torchvision.datasets.Food101(
        root, split='train', download=True, transform=TRAIN_TFMS
    )

    testset = torchvision.datasets.Food101(
        root, split='test', download=True, transform=TEST_TFMS
    )

    return trainset, testset

def get_oxfordiitpet_dataset(root: str):

    trainset = torchvision.datasets.OxfordIIITPet(
        root, split='trainval', download=True, transform=TRAIN_TFMS, target_types='category'
    )

    testset = torchvision.datasets.OxfordIIITPet(
        root, split='test', download=True, transform=TEST_TFMS, target_types='category'
    )

    return trainset, testset

def get_cinic10_dataset(root: str):
    trainset = torchvision.datasets.ImageFolder(
        os.path.join(root, 'train'),
        TRAIN_TFMS
    )
    testset = torchvision.datasets.ImageFolder(
        os.path.join(root, 'test'),
        TEST_TFMS
    )
    return trainset, testset

def get_tinynet_dataset(root: str):

    trainset = TinyImageNet(root, split='train', transform=TRAIN_TFMS)

    testset = TinyImageNet(root, split='val', transform=TEST_TFMS)

    return trainset, testset

class TinyImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.data = []
        self.labels = []
        self.label_map = self._create_label_map()

        self._load_data()

    def _create_label_map(self):
        label_map = {}
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                label_map[line.strip()] = idx
        return label_map

    def _load_data(self):
        if self.split == 'train':
            for label in self.label_map.keys():
                label_dir = os.path.join(self.root, 'train', label, 'images')
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(self.label_map[label])
        elif self.split == 'val':
            val_img_dir = os.path.join(self.root, 'val', 'images')
            with open(os.path.join(self.root, 'val', 'val_annotations.txt'), 'r') as f:
                val_annotations = f.readlines()
            for annotation in val_annotations:
                parts = annotation.split('\t')
                img_name = parts[0]
                img_path = os.path.join(val_img_dir, img_name)
                label = parts[1]
                self.data.append(img_path)
                self.labels.append(self.label_map[label])
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label
    
class ImageNetDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Args:
            split (str): The split of the dataset to use, e.g., 'train', 'validation'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.dataset = load_dataset("ILSVRC/imagenet-1k", split=split, trust_remote_code=True, use_auth_token=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        # Convert grayscale to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_imagenet1k_dataset(root: str):
    trainset = ImageNetDataset(split='train', transform=TRAIN_TFMS)
    testset = ImageNetDataset(split='validation', transform=TEST_TFMS)

    return trainset, testset