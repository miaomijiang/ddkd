import yaml
from pathlib import Path
from typing import Dict

class ConfigLoader:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)

    def load(self, config_name: str) -> Dict:
        config_path = self._resolve_config_path(config_name)

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 处理基类继承
        if '_base_' in config:
            base_config_name = config.pop('_base_')
            base_config = self.load(base_config_name)
            config = self._merge_configs(base_config, config)

        return config

    def _resolve_config_path(self, config_name: str) -> Path:
        config_path = Path(config_name)
        if config_path.exists():
            return config_path

        config_path = self.config_dir / config_name
        if config_path.exists():
            return config_path

        if not config_name.endswith('.yaml'):
            config_path = self.config_dir / (config_name + '.yaml')
            if config_path.exists():
                return config_path

        raise FileNotFoundError(f"配置文件未找到: {config_name}")

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result