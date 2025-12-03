import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config.config import (
    DATASET_META, DATASET_NAME, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    DATA_SOURCE, SYNTHETIC_DATA_PATH, DATA_DISTILL_BATCH_SIZE, DEVICE
)

# 校验数据集配置是否有效
if DATASET_NAME not in DATASET_META.keys():
    raise ValueError(
        f"无效的DATASET_NAME：{DATASET_NAME}！\n"
        f"支持的数据集：{list(DATASET_META.keys())}\n"
        f"请在config/config.py的DATASET_META中添加对应配置"
    )
DATASET_CONFIG = DATASET_META[DATASET_NAME]  # 当前数据集的配置


def get_data_loaders(stage="DISTILL"):
    """
    完全配置驱动的数据加载器：无任何硬编码数据集判断
    :param stage: 阶段标识：DISTILL（数据蒸馏）/ KD（知识蒸馏）
    :return: train_loader, val_loader, test_loader, num_classes
    """
    # ==================== 通用数据预处理 ====================
    # 训练集预处理（数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(IMAGE_SIZE, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集/测试集预处理（无增强）
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==================== 加载原始数据集（基础数据） ====================
    # 公开数据集（torchvision直接加载，如CIFAR10/CIFAR100/MNIST）
    if DATASET_CONFIG["is_public"]:
        # 动态导入torchvision中的数据集类（无硬编码）
        dataset_class = getattr(datasets, DATASET_NAME)

        # 训练集（固定train=True）
        train_dataset_original = dataset_class(
            root=DATASET_CONFIG["root"],
            train=True,
            download=DATASET_CONFIG["download"],
            transform=train_transform
        )

        # 测试集（固定train=False）
        test_dataset = dataset_class(
            root=DATASET_CONFIG["root"],
            train=False,
            download=DATASET_CONFIG["download"],
            transform=val_test_transform
        )

        # 验证集：根据配置判断是否复用测试集
        if DATASET_CONFIG["val_reuse_test"]:
            val_dataset = test_dataset  # CIFAR10/MNIST等无单独验证集
        else:
            # 若公开数据集有单独验证集（如ImageNet），可在这里扩展逻辑
            val_dataset = dataset_class(
                root=DATASET_CONFIG["root"],
                train=False,  # 需根据数据集实际情况调整（如split="val"）
                download=DATASET_CONFIG["download"],
                transform=val_test_transform
            )

        # 类别数：从配置中读取（公开数据集固定）
        num_classes = DATASET_CONFIG["num_classes"]

    # 私有数据集（按文件夹结构加载：train/val/test子文件夹）
    else:
        # 训练集（train子文件夹）
        train_dataset_original = datasets.ImageFolder(
            root=os.path.join(DATASET_CONFIG["root"], "train"),
            transform=train_transform
        )

        # 验证集（val子文件夹）
        val_dataset = datasets.ImageFolder(
            root=os.path.join(DATASET_CONFIG["root"], "val"),
            transform=val_test_transform
        )

        # 测试集（test子文件夹）
        test_dataset = datasets.ImageFolder(
            root=os.path.join(DATASET_CONFIG["root"], "test"),
            transform=val_test_transform
        )

        # 类别数：自动从文件夹识别（私有数据集配置中num_classes=None）
        num_classes = len(train_dataset_original.classes)
        # 更新配置中的类别数（后续复用）
        DATASET_META[DATASET_NAME]["num_classes"] = num_classes

    # ==================== 根据阶段和配置选择训练集来源 ====================
    # 1. 数据蒸馏阶段（DISTILL）：固定用原始训练集（生成合成数据）
    if stage == "DISTILL":
        train_dataset = train_dataset_original
        current_batch_size = DATA_DISTILL_BATCH_SIZE
        train_data_desc = f"原始{DATASET_NAME}（{len(train_dataset)}张）"

    # 2. 知识蒸馏阶段（KD）：按DATA_SOURCE选择原始/合成数据
    elif stage == "KD":
        if DATA_SOURCE == "ORIGINAL":
            train_dataset = train_dataset_original
            current_batch_size = BATCH_SIZE
            train_data_desc = f"原始{DATASET_NAME}（{len(train_dataset)}张）"

        elif DATA_SOURCE == "SYNTHETIC":
            # 验证合成数据集是否存在
            if not os.path.exists(SYNTHETIC_DATA_PATH):
                raise FileNotFoundError(
                    f"合成数据集不存在！\n"
                    f"请设置RUN_DATA_DISTILL=True生成，或修改DATA_SOURCE=ORIGINAL"
                )
            # 加载合成数据
            data_dict = torch.load(SYNTHETIC_DATA_PATH, map_location=DEVICE)
            synthetic_data = data_dict["synthetic_data"]
            synthetic_labels = data_dict["synthetic_labels"]
            # 包装成Dataset
            train_dataset = TensorDataset(synthetic_data, synthetic_labels)
            current_batch_size = BATCH_SIZE
            train_data_desc = f"合成数据（{len(train_dataset)}张，IPC={len(train_dataset) // num_classes}）"

        else:
            raise ValueError(
                f"无效的DATA_SOURCE：{DATA_SOURCE}！\n"
                f"支持的来源：ORIGINAL/SYNTHETIC，请在config中修改"
            )

    else:
        raise ValueError(
            f"无效的stage：{stage}！\n"
            f"仅支持DISTILL（数据蒸馏）/ KD（知识蒸馏）"
        )

    # ==================== 生成DataLoader ====================
    train_loader = DataLoader(
        train_dataset,
        batch_size=current_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True  # 避免最后一个batch尺寸不统一
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 打印加载信息（统一格式）
    print(f"📊 阶段：{stage} | 数据集：{DATASET_NAME}")
    print(f"📊 训练集：{train_data_desc}")
    print(f"📊 验证集：{len(val_dataset)}张 | 测试集：{len(test_dataset)}张")
    print(f"📊 Batch Size：训练集={current_batch_size} | 验证集/测试集={BATCH_SIZE * 2}")
    print(f"📊 类别数：{num_classes}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, num_classes