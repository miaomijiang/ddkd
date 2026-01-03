"""
数据导入模块 - 自动获取公开数据集
"""
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Tuple, Dict
from PIL import Image


class SimpleCustomDataset(Dataset):
    """简单的自定义数据集类"""

    def __init__(self, data_dir: str, transform=None, split='train', split_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split

        # 收集数据
        self.samples = []
        self.labels = []
        self.class_names = []

        # 获取所有类别（假设按文件夹组织）
        self.class_names = sorted([d for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))])

        # 为每个类别收集图片
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)

            # 获取所有图片文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                image_files.extend([f for f in os.listdir(class_dir)
                                  if f.lower().endswith(ext)])

            # 划分训练/测试集
            split_idx = int(len(image_files) * split_ratio)

            if split == 'train':
                selected_files = image_files[:split_idx]
            else:  # 'test'
                selected_files = image_files[split_idx:]

            # 添加到数据集
            for file_name in selected_files:
                self.samples.append(os.path.join(class_dir, file_name))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class DatasetLoader:
    """数据集加载器"""

    SUPPORTED_DATASETS = {
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "custom": SimpleCustomDataset,  # 添加自定义数据集支持
    }

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config['name']

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        self._set_dataset_params()

    def _set_dataset_params(self):
        """设置数据集参数"""
        if self.dataset_name in ["mnist", "fashionmnist"]:
            self.num_channels = 1
            self.mean = (0.1307,)
            self.std = (0.3081,)
            self.image_size = 28
            self.num_classes = 10
        elif self.dataset_name == "cifar10":
            self.num_channels = 3
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
            self.image_size = 32
            self.num_classes = 10
        elif self.dataset_name == "cifar100":
            self.num_channels = 3
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
            self.image_size = 32
            self.num_classes = 100
        elif self.dataset_name == "custom":  # 自定义数据集参数
            self.num_channels = 3
            self.mean = (0.5, 0.5, 0.5)  # 默认值
            self.std = (0.5, 0.5, 0.5)   # 默认值
            self.image_size = self.config.get('image_size', 224)  # 可配置
            self.num_classes = 0  # 将在加载时确定

    def get_transforms(self, augment: bool = True) -> transforms.Compose:
        """获取数据转换"""
        transform_list = []

        if augment:
            if self.dataset_name in ["cifar10", "cifar100"]:
                transform_list.extend([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ])
            elif self.dataset_name == "custom":
                # 自定义数据集的数据增强
                transform_list.extend([
                    transforms.RandomResizedCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                ])

        # 调整图像大小（主要针对自定义数据集）
        if self.dataset_name == "custom":
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        return transforms.Compose(transform_list)

    def load(self) -> Tuple:
        """加载数据集"""
        train_transform = self.get_transforms(augment=True)
        test_transform = self.get_transforms(augment=False)

        dataset_class = self.SUPPORTED_DATASETS[self.dataset_name]

        if self.dataset_name == "custom":
            # 自定义数据集的特殊处理
            split_ratio = self.config.get('split_ratio', 0.8)

            train_dataset = dataset_class(
                data_dir=self.config['data_dir'],
                transform=train_transform,
                split='train',
                split_ratio=split_ratio
            )

            test_dataset = dataset_class(
                data_dir=self.config['data_dir'],
                transform=test_transform,
                split='test',
                split_ratio=split_ratio
            )

            # 获取实际的类别数
            self.num_classes = len(train_dataset.class_names)

        else:
            # 原有公开数据集的加载方式
            train_dataset = dataset_class(
                root=self.config['data_dir'],
                train=True,
                download=self.config['download'],
                transform=train_transform
            )

            test_dataset = dataset_class(
                root=self.config['data_dir'],
                train=False,
                download=self.config['download'],
                transform=test_transform
            )

        return train_dataset, test_dataset

    def get_dataloaders(self) -> Tuple:
        """获取数据加载器"""
        train_dataset, test_dataset = self.load()

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        return train_loader, test_loader

    def get_num_classes(self) -> int:
        """获取类别数"""
        return self.num_classes