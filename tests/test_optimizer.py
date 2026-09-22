import unittest

import torch

from engine.trainer import _make_optimizer
from utils.schedulers import UpdateLRScheduler


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.decoder = torch.nn.Linear(4, 2)


class DifferentialLRTests(unittest.TestCase):
    def test_parameter_groups_are_complete_disjoint_and_named(self):
        model = TinySegmenter()
        config = {
            "optimizer": {
                "name": "sgd",
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr_scale": "none",
                "differential_lr": {
                    "encoder_lr": 0.001,
                    "decoder_lr": 0.01,
                },
            }
        }
        optimizer, _ = _make_optimizer(config, model, effective_batch=16)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["encoder", "decoder"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.001, 0.01],
        )
        grouped_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(
            set(grouped_ids),
            {id(parameter) for parameter in model.parameters()},
        )

    def test_scheduler_resume_restores_both_group_lrs(self):
        model = TinySegmenter()
        config = {
            "optimizer": {
                "name": "sgd",
                "lr": 0.01,
                "lr_scale": "none",
                "differential_lr": {
                    "encoder_lr": 0.001,
                    "decoder_lr": 0.01,
                },
            }
        }
        optimizer, _ = _make_optimizer(config, model, effective_batch=16)
        scheduler = UpdateLRScheduler(
            optimizer, 1000, "poly", warmup_updates=100
        )
        scheduler.step(350)
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        resumed_model = TinySegmenter()
        resumed_optimizer, _ = _make_optimizer(config, resumed_model, effective_batch=16)
        resumed_scheduler = UpdateLRScheduler(
            resumed_optimizer, 1000, "poly", warmup_updates=100
        )
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        self.assertEqual(
            [group["lr"] for group in resumed_optimizer.param_groups],
            [group["lr"] for group in optimizer.param_groups],
        )


if __name__ == "__main__":
    unittest.main()
