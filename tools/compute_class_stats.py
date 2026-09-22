"""兼容入口：执行项目根目录已有的类别统计工具。"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "compute_class_stats.py"), run_name="__main__")
