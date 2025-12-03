import torch
import numpy as np
from config.config import *
from utils.data_loader import get_data_loaders
from models import StudentModel


def evaluate():
    # 1. 加载数据和模型
    _, _, test_loader = get_data_loaders()
    student_model = StudentModel(num_classes=len(test_loader.dataset.classes)).to(DEVICE)

    # 加载最佳模型权重
    model_path = os.path.join(SAVE_DIR, "best_student_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("未找到最佳模型，请先训练！")
    student_model.load_state_dict(torch.load(model_path))
    student_model.eval()  # 评估模式

    # 2. 评估指标
    correct = 0
    total = 0
    inference_times = []
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0

    print("=" * 60)
    print("开始模型评估（测试集）...")
    print(f"模型路径：{model_path}")
    print(f"测试集样本数：{len(test_loader.dataset)}")
    print("=" * 60)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 推理时间统计
            start = time.time()
            outputs = student_model(images)
            inference_times.append(time.time() - start)

            # 损失和准确率统计
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 3. 输出结果
    avg_loss = total_loss / len(test_loader)
    avg_acc = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒

    print("\n评估结果：")
    print(f"测试集损失：{avg_loss:.4f}")
    print(f"测试集准确率：{avg_acc:.2f}%")
    print(f"平均推理时间：{avg_inference_time:.2f}ms/样本")
    print(f"每秒推理帧数（FPS）：{1000 / avg_inference_time:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import time

    evaluate()