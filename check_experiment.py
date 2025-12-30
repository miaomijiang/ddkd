import json
import os
from pathlib import Path

print('🔍 检查实验结果...')
print('='*60)

# 1. 检查实验目录
exp_dir = Path('experiments')
if exp_dir.exists():
    experiments = list(exp_dir.iterdir())
    if experiments:
        print(f'📁 发现 {len(experiments)} 个实验:')
        for exp in experiments:
            print(f'   - {exp.name}')

        # 获取最新实验（按修改时间排序）
        latest_exp = sorted(experiments, key=lambda x: x.stat().st_mtime)[-1]
        print(f'\n📊 分析最新实验: {latest_exp.name}')

        # 检查实验目录下的文件
        files = list(latest_exp.iterdir())
        print(f'   包含 {len(files)} 个文件:')
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f'   - {f.name} ({size_kb:.1f} KB)')

        # 读取results.json结果文件
        results_file = latest_exp / 'results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            print('\n🎯 关键结果:')
            print(f'   数据集: {results["config"]["dataset"]["name"]}')
            print(f'   教师模型: {results["config"]["teacher"]["name"]}')
            print(f'   学生模型: {results["config"]["student"]["name"]}')
            print(f'   训练轮数: {results["config"]["training"]["epochs"]}')
            print(f'   训练时间: {results["training_time"]:.1f} 秒')

            if 'baseline' in results:
                print(f'\n📈 基准模型:')
                print(f'   Top-1准确率: {results["baseline"]["accuracy_top1"]:.2f}%')
                print(f'   Top-5准确率: {results["baseline"]["accuracy_top5"]:.2f}%')

            if 'distilled' in results:
                print(f'\n🔥 知识蒸馏模型:')
                print(f'   Top-1准确率: {results["distilled"]["accuracy_top1"]:.2f}%')
                print(f'   Top-5准确率: {results["distilled"]["accuracy_top5"]:.2f}%')

            if 'comparison' in results:
                print(f'\n📊 性能对比:')
                print(f'   Top-1提升: {results["comparison"]["accuracy_top1_improvement"]:+.2f}%')
                print(f'   Top-5提升: {results["comparison"]["accuracy_top5_improvement"]:+.2f}%')
                print(f'   模型大小比: {results["comparison"]["size_ratio"]:.2f}x')
                print(f'   推理速度比: {results["comparison"]["inference_time_ratio"]:.2f}x')
        else:
            print('⚠️  未找到 results.json 文件')
    else:
        print('⚠️  实验目录为空')
else:
    print('⚠️  实验目录不存在')

print('\n' + '='*60)

# 2. 检查合成图像
synth_dir = Path('synthetic_images')
if synth_dir.exists():
    images = list(synth_dir.glob('*.png'))
    print(f'🖼️  合成图像: {len(images)} 张')
    if images:
        # 显示最新5张
        latest_images = sorted(images, key=lambda x: x.stat().st_mtime)[-5:]
        print('   最新5张:')
        for img in latest_images:
            size_kb = img.stat().st_size / 1024
            print(f'   - {img.name} ({size_kb:.1f} KB)')
else:
    print('🖼️  未找到合成图像目录')

# 3. 检查模型文件
model_files = list(Path('.').glob('*.pth'))
print(f'\n💾 模型文件: {len(model_files)} 个')
for model in model_files:
    size_mb = model.stat().st_size / (1024*1024)
    print(f'   - {model.name} ({size_mb:.1f} MB)')

print('='*60)