"""按 optimizer update 而非 epoch 更新的学习率调度器。"""

from __future__ import annotations

import math


class UpdateLRScheduler:
    def __init__(
        self,
        optimizer,
        max_updates: int,
        name: str,
        power: float = 0.9,
        warmup_updates: int = 0,
        warmup_start_factor: float = 0.01,
    ):
        if max_updates <= 0:
            raise ValueError("max_updates 必须大于 0。")
        if name not in {"poly", "cosine", "constant"}:
            raise ValueError("scheduler.name 仅支持 poly、cosine、constant。")
        if warmup_updates < 0 or warmup_updates >= max_updates:
            raise ValueError("warmup_updates 必须位于 [0, max_updates) 范围内。")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor 必须位于 (0, 1]。")
        self.optimizer = optimizer
        self.max_updates = int(max_updates)
        self.name = name
        self.power = float(power)
        self.warmup_updates = int(warmup_updates)
        self.warmup_start_factor = float(warmup_start_factor)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.update = 0
        self._apply_lrs()

    def _factor(self) -> float:
        if self.warmup_updates and self.update < self.warmup_updates:
            progress = self.update / self.warmup_updates
            return self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * progress

        decay_updates = self.max_updates - self.warmup_updates
        decay_progress = (
            self.update - self.warmup_updates
        ) / decay_updates
        progress = min(max(decay_progress, 0.0), 1.0)
        if self.name == "poly":
            return (1.0 - progress) ** self.power
        if self.name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def _apply_lrs(self) -> None:
        factor = self._factor()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def step(self, completed_updates: int | None = None) -> None:
        if completed_updates is not None:
            self.update = int(completed_updates)
        else:
            self.update += 1
        if not 0 <= self.update <= self.max_updates:
            raise ValueError("completed_updates 必须位于 [0, max_updates]。")
        self._apply_lrs()

    def state_dict(self) -> dict:
        return {
            "max_updates": self.max_updates,
            "name": self.name,
            "power": self.power,
            "warmup_updates": self.warmup_updates,
            "warmup_start_factor": self.warmup_start_factor,
            "base_lrs": self.base_lrs,
            "update": self.update,
        }

    def load_state_dict(self, state: dict) -> None:
        # 兼容加入 warmup 之前保存的 baseline checkpoint；它们等价于关闭 warmup。
        state = dict(state)
        state.setdefault("warmup_updates", 0)
        state.setdefault("warmup_start_factor", 0.01)
        for key in (
            "max_updates", "name", "power", "warmup_updates",
            "warmup_start_factor", "base_lrs",
        ):
            if state[key] != getattr(self, key):
                raise ValueError(f"scheduler resume 参数不一致：{key}")
        self.update = int(state["update"])
        self.step(self.update)
