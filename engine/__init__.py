"""训练与验证流程。"""

from .evaluator import evaluate_full_validation
from .trainer import run_training

__all__ = ["evaluate_full_validation", "run_training"]
