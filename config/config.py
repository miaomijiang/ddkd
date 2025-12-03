import torch
import os

# ==================== 基础配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # 固定随机种子，保证复现性
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models/saved_weights", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# ==================== 数据集元配置（核心：完全配置驱动，无硬编码） ====================
# 新增：定义所有支持的数据集属性，新增数据集只需在这里添加配置
DATASET_META = {
    "CIFAR10": {
        "is_public": True,  # 是否为公开数据集（torchvision直接加载）
        "num_classes": 10,  # 固定类别数
        "val_reuse_test": True,  # 验证集是否复用测试集（CIFAR10无单独验证集）
        "root": "./data",  # 数据存储根目录
        "download": True  # 是否自动下载（公开数据集专用）
    },
    "CIFAR100": {
        "is_public": True,
        "num_classes": 100,
        "val_reuse_test": True,
        "root": "./data",
        "download": True
    },
    "MNIST": {
        "is_public": True,
        "num_classes": 10,
        "val_reuse_test": True,
        "root": "./data",
        "download": True
    },
    "PRIVATE": {
        "is_public": False,  # 私有数据集（按文件夹结构加载）
        "num_classes": None,  # 自动从文件夹识别类别数
        "val_reuse_test": False,  # 私有数据集需单独划分val/test集
        "root": "./data/private",  # 私有数据集根目录（train/val/test子文件夹）
        "download": False  # 私有数据集无需下载
    }
}

# 选择当前使用的数据集（核心参数：只需修改这里切换数据集）
DATASET_NAME = "CIFAR10"  # 可选：CIFAR10/CIFAR100/MNIST/PRIVATE

# ==================== 数据集通用配置（统一控制） ====================
IMAGE_SIZE = 128  # 图像尺寸（所有数据集统一适配）
NUM_WORKERS = 0  # 4GB GPU建议0，避免多线程显存占用
DATA_SOURCE = "SYNTHETIC"  # 训练集来源：ORIGINAL（原始）/ SYNTHETIC（合成）
SYNTHETIC_DATA_PATH = "./data/synthetic_dataset.pt"  # 合成数据集路径

# ==================== 数据蒸馏（Dataset Distillation）配置 ====================
RUN_DATA_DISTILL = True  # 是否先执行数据蒸馏（生成合成数据）
DATA_DISTILL_EPOCHS = 2000  # 数据蒸馏训练轮数
DATA_DISTILL_LR = 1e-3  # 数据蒸馏学习率
IPC = 50  # 每类合成图像数（CIFAR10→50*10=500张）
DATA_DISTILL_BATCH_SIZE = 2  # 数据蒸馏阶段batch size（4GB GPU适配）

# ==================== 知识蒸馏（Knowledge Distillation）配置 ====================
BATCH_SIZE = 4  # 知识蒸馏阶段batch size
LEARNING_RATE = 3e-3
EPOCHS = 80
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"  # cosine/step
STEP_SIZE = 20  # StepLR调度器的步长
GAMMA = 0.1  # StepLR调度器的衰减系数

# 损失权重配置
TEMPERATURE = 4.0  # 蒸馏温度
DISTILL_WEIGHT = 0.7  # 软化蒸馏损失权重
GRADIENT_MATCH_WEIGHT = 0.05  # 梯度匹配损失权重

# ==================== 模型保存配置 ====================
SAVE_DIR = "./models/saved_weights"
SAVE_FREQ = 10  # 每10轮保存一次模型
LOG_DIR = "./logs"  # Tensorboard日志目录