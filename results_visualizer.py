# results_visualizer.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_results():
    # 找到最新实验结果
    exp_dir = Path("experiments")
    experiments = list(exp_dir.iterdir())
    if not experiments:
        print("没有找到实验结果")
        return

    latest_exp = sorted(experiments, key=lambda x: x.stat().st_mtime)[-1]
    results_file = latest_exp / "results.json"

    if not results_file.exists():
        print(f"在 {latest_exp} 中未找到results.json")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 准确率对比
    ax1 = axes[0, 0]
    metrics = ['Top-1 Accuracy', 'Top-5 Accuracy']
    if 'baseline' in results and 'distilled' in results:
        baseline_acc = [
            results['baseline']['accuracy_top1'],
            results['baseline']['accuracy_top5']
        ]
        distilled_acc = [
            results['distilled']['accuracy_top1'],
            results['distilled']['accuracy_top5']
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, baseline_acc, width, label='Baseline', color='skyblue')
        bars2 = ax1.bar(x + width / 2, distilled_acc, width, label='Distilled', color='lightcoral')

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. 性能提升
    ax2 = axes[0, 1]
    if 'comparison' in results:
        improvements = ['Top-1', 'Top-5']
        values = [
            results['comparison']['accuracy_top1_improvement'],
            results['comparison']['accuracy_top5_improvement']
        ]

        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax2.bar(improvements, values, color=colors)

        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Knowledge Distillation Improvement')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

    # 3. 训练信息
    ax3 = axes[1, 0]
    training_info = {
        'Training Time (s)': results['training_time'],
        'Epochs': results['config']['training']['epochs'],
        'IPC': results['config']['data_distillation']['ipc']
    }

    names = list(training_info.keys())
    values = list(training_info.values())

    bars = ax3.barh(names, values, color='lightgreen')
    ax3.set_xlabel('Value')
    ax3.set_title('Training Configuration')
    ax3.grid(True, alpha=0.3)

    # 4. 模型信息
    ax4 = axes[1, 1]
    if 'distilled' in results:
        model_info = {
            'Parameters': results['distilled']['total_parameters'] / 1e6,  # 百万
            'Size (MB)': results['distilled']['model_size_mb'],
            'Inference Time (ms)': results['distilled']['inference_time_per_sample'] * 1000
        }

        names = list(model_info.keys())
        values = list(model_info.values())

        bars = ax4.bar(names, values, color='orange')
        ax4.set_ylabel('Value')
        ax4.set_title('Model Characteristics')
        ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Knowledge Distillation Experiment: {results['config']['dataset']['name']}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图表
    output_path = latest_exp / "results_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 可视化图表已保存: {output_path}")
    plt.show()


if __name__ == "__main__":
    visualize_results()