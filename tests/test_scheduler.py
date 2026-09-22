import unittest

import torch

from utils.schedulers import UpdateLRScheduler


class SchedulerTests(unittest.TestCase):
    def _optimizer(self):
        first = torch.nn.Parameter(torch.tensor(1.0))
        second = torch.nn.Parameter(torch.tensor(2.0))
        return torch.optim.SGD(
            [
                {"params": [first], "lr": 0.001, "name": "encoder"},
                {"params": [second], "lr": 0.01, "name": "decoder"},
            ]
        )

    def test_linear_warmup_uses_optimizer_updates(self):
        optimizer = self._optimizer()
        scheduler = UpdateLRScheduler(
            optimizer,
            max_updates=1000,
            name="poly",
            power=0.9,
            warmup_updates=100,
            warmup_start_factor=0.01,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00001)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.0001)
        scheduler.step(50)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.000505)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.00505)
        scheduler.step(100)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.01)

    def test_all_parameter_groups_decay_with_fixed_ratio(self):
        optimizer = self._optimizer()
        scheduler = UpdateLRScheduler(optimizer, 100, "poly", power=0.9)
        scheduler.step(50)
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"],
            10.0,
        )

    def test_resume_restores_warmup_and_poly_position(self):
        first_optimizer = self._optimizer()
        first = UpdateLRScheduler(
            first_optimizer, 1000, "poly", warmup_updates=100
        )
        first.step(437)
        state = first.state_dict()

        resumed_optimizer = self._optimizer()
        resumed = UpdateLRScheduler(
            resumed_optimizer, 1000, "poly", warmup_updates=100
        )
        resumed.load_state_dict(state)
        self.assertEqual(resumed.update, 437)
        self.assertEqual(
            [group["lr"] for group in resumed_optimizer.param_groups],
            [group["lr"] for group in first_optimizer.param_groups],
        )

    def test_legacy_state_is_accepted_when_warmup_disabled(self):
        optimizer = self._optimizer()
        scheduler = UpdateLRScheduler(optimizer, 100, "poly")
        legacy = scheduler.state_dict()
        legacy.pop("warmup_updates")
        legacy.pop("warmup_start_factor")
        scheduler.load_state_dict(legacy)


if __name__ == "__main__":
    unittest.main()
