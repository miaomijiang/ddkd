import torch
import os

# ==================== 基础配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # 固定随机种子，保证复现性
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models/saved_weights", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# ==================== 数据集配置（统一控制来源） ====================
DATASET_TYPE = "CIFAR10"  # 数据集类型：CIFAR10/PRIVATE
DATASET_ROOT = "./data"  # 原始数据集根目录
IMAGE_SIZE = 128  # 图像尺寸（统一适配原始/合成数据）
NUM_WORKERS = 0  # 4GB GPU建议0，避免多线程显存占用

# 新增：数据集来源控制（核心参数）
DATA_SOURCE = "SYNTHETIC"  # 可选：ORIGINAL（原始数据）/ SYNTHETIC（合成数据）
SYNTHETIC_DATA_PATH = "./data/synthetic_dataset.pt"  # 合成数据集路径（与数据蒸馏保存路径一致）

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

# ==================== 数据集类别数配置（公开数据集预设） ====================
DATASET_NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "PRIVATE": None  # 私有数据集自动从文件夹识别
}