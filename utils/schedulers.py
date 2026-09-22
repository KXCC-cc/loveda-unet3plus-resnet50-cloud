"""按 optimizer update 而非 epoch 更新的学习率调度器。"""

from __future__ import annotations

import math


class UpdateLRScheduler:
    def __init__(self, optimizer, max_updates: int, name: str, power: float = 0.9):
        if max_updates <= 0:
            raise ValueError("max_updates 必须大于 0。")
        if name not in {"poly", "cosine", "constant"}:
            raise ValueError("scheduler.name 仅支持 poly、cosine、constant。")
        self.optimizer = optimizer
        self.max_updates = int(max_updates)
        self.name = name
        self.power = float(power)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.update = 0

    def step(self, completed_updates: int | None = None) -> None:
        if completed_updates is not None:
            self.update = int(completed_updates)
        else:
            self.update += 1
        progress = min(max(self.update / self.max_updates, 0.0), 1.0)
        if self.name == "poly":
            factor = (1.0 - progress) ** self.power
        elif self.name == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            factor = 1.0
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self) -> dict:
        return {
            "max_updates": self.max_updates,
            "name": self.name,
            "power": self.power,
            "base_lrs": self.base_lrs,
            "update": self.update,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("max_updates", "name", "power", "base_lrs"):
            if state[key] != getattr(self, key):
                raise ValueError(f"scheduler resume 参数不一致：{key}")
        self.update = int(state["update"])
        self.step(self.update)
