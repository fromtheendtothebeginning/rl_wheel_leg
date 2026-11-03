"""读取 ``config/project.ini``，把路径和仿真参数从代码里挪出来。

为什么用 ini 而不是 yaml / toml：MuJoCo 那个 venv 是 Python 3.10（没有 tomllib），
装 PyYAML 又给接手的人多一个依赖；configparser 是标准库，3.10 和 3.12 都能用。

路径类配置支持 ``${HOME}`` 展开；相对路径按仓库根解析，所以从哪个目录运行都一样。
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "project.ini"


class Config:
    """比 configparser 多一层类型转换和路径解析。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG
        if not self.path.exists():
            raise FileNotFoundError(f"找不到配置文件: {self.path}")
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(self.path, encoding="utf-8")
        self._parser = parser

    def get(self, section: str, key: str, fallback: str | None = None) -> str:
        if self._parser.has_option(section, key):
            return self._parser.get(section, key).strip()
        if fallback is not None:
            return fallback
        raise KeyError(f"{self.path} 缺少 [{section}] {key}")

    def float(self, section: str, key: str, fallback: float | None = None) -> float:
        return float(self.get(section, key, None if fallback is None else str(fallback)))

    def int(self, section: str, key: str, fallback: int | None = None) -> int:
        return int(self.get(section, key, None if fallback is None else str(fallback)))

    def bool(self, section: str, key: str, fallback: bool | None = None) -> bool:
        raw = self.get(section, key, None if fallback is None else str(fallback))
        return raw.lower() in {"1", "true", "yes", "on"}

    def path_of(self, section: str, key: str, fallback: str | None = None) -> Path:
        raw = os.path.expandvars(os.path.expanduser(self.get(section, key, fallback)))
        resolved = Path(raw)
        return resolved if resolved.is_absolute() else (REPO_ROOT / resolved)

    def robot_dir(self) -> Path:
        return self.path_of("paths", "robot_dir")

    def out_dir(self) -> Path:
        return self.path_of("paths", "out_dir")


def load(path: str | Path | None = None) -> Config:
    return Config(path)
