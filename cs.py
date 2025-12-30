# test_config.py
import yaml
from pathlib import Path

config_path = Path("configs/cifar10.yaml")
print(f"配置文件路径: {config_path.absolute()}")
print(f"文件是否存在: {config_path.exists()}")

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print("文件内容前200字符:")
        print(content[:200])

        # 尝试解析
        f.seek(0)  # 重置文件指针
        try:
            config = yaml.safe_load(f)
            print(f"\nYAML解析成功! 配置类型: {type(config)}")
            if config:
                print("配置内容预览:", config)
            else:
                print("警告: 配置文件内容为空或全为注释!")
        except yaml.YAMLError as e:
            print(f"\nYAML解析错误: {e}")