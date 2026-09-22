import tempfile
import unittest
from pathlib import Path

import torch

from engine.trainer import _prepare_output_directory, _save_epoch_checkpoint
from utils.checkpoint import load_experiment_checkpoint
from utils.device import create_grad_scaler
from utils.schedulers import UpdateLRScheduler


class RunDirectorySafetyTests(unittest.TestCase):
    def test_new_run_rejects_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "best_model.pth").touch()

            with self.assertRaises(FileExistsError):
                _prepare_output_directory(output_dir, resume=None)

    def test_resume_checkpoint_must_belong_to_output_directory(self):
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                output_dir = Path(first_directory)
                checkpoint = Path(second_directory) / "last_checkpoint.pth"
                checkpoint.touch()

                with self.assertRaises(ValueError):
                    _prepare_output_directory(output_dir, resume=checkpoint)

    @staticmethod
    def _training_objects():
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = UpdateLRScheduler(
            optimizer, max_updates=100, name="poly", power=0.9
        )
        scaler = create_grad_scaler(False)
        generator = torch.Generator().manual_seed(42)
        config = {
            "model": {"type": "test"},
            "data": {"root": "test"},
            "loss": {"ce_weight": 1.0},
            "optimizer": {"name": "sgd", "lr": 0.01},
            "scheduler": {"name": "poly", "power": 0.9},
            "training": {"max_iters": 100},
        }
        return model, optimizer, scheduler, scaler, generator, config

    def test_non_validation_epoch_saves_complete_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            model, optimizer, scheduler, scaler, generator, config = (
                self._training_objects()
            )
            scheduler.step(17)
            _save_epoch_checkpoint(
                output_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                generator,
                epoch=3,
                micro_step=136,
                optimizer_step=17,
                best_miou=0.45,
                save_best=False,
            )
            checkpoint_path = output_dir / "last_checkpoint.pth"
            self.assertTrue(checkpoint_path.is_file())
            self.assertFalse((output_dir / "best_model.pth").exists())
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            required = {
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "epoch",
                "micro_step",
                "optimizer_step",
                "best_miou",
                "rng_state",
                "config",
            }
            self.assertTrue(required.issubset(checkpoint))
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["optimizer_step"], 17)

    def test_resume_restores_optimizer_step_and_scheduler_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            model, optimizer, scheduler, scaler, generator, config = (
                self._training_objects()
            )
            scheduler.step(23)
            _save_epoch_checkpoint(
                output_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                generator,
                epoch=4,
                micro_step=184,
                optimizer_step=23,
                best_miou=0.46,
            )

            resumed = self._training_objects()
            (
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_scaler,
                _,
                _,
            ) = resumed
            state = load_experiment_checkpoint(
                output_dir / "last_checkpoint.pth",
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                resumed_scaler,
                current_config=config,
            )
            self.assertEqual(state["epoch"], 4)
            self.assertEqual(state["epoch"] + 1, 5)
            self.assertEqual(state["optimizer_step"], 23)
            self.assertEqual(resumed_scheduler.update, 23)
            self.assertEqual(
                resumed_scheduler.state_dict(), scheduler.state_dict()
            )
            self.assertEqual(
                resumed_optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["lr"],
            )


if __name__ == "__main__":
    unittest.main()
