import unittest
from pathlib import Path

from utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class CloudConfigTests(unittest.TestCase):
    def _load(self, name):
        return load_config(ROOT / "configs" / "cloud" / name)

    def test_baseline_remains_clean(self):
        config = self._load("resnet50_24gb.yaml")
        self.assertEqual(config["data"]["train_scales"], [1.0])
        self.assertEqual(config["data"]["class_aware_crop"]["probability"], 0.0)
        self.assertEqual(config["loss"]["focal_weight"], 0.0)
        self.assertIsNone(config["optimizer"]["differential_lr"])
        self.assertEqual(config["scheduler"].get("warmup_updates", 0), 0)

    def test_multiscale_and_classaware_combination(self):
        config = self._load("resnet50_24gb_multiscale_classaware.yaml")
        self.assertEqual(config["data"]["train_scales"], [0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
        self.assertEqual(config["data"]["crop_size"], [512, 512])
        self.assertEqual(config["data"]["class_aware_crop"]["probability"], 0.5)

    def test_focal_ablation_does_not_duplicate_full_ce(self):
        config = self._load("resnet50_24gb_focal.yaml")
        self.assertEqual(config["loss"]["ce_weight"], 0.5)
        self.assertEqual(config["loss"]["focal_weight"], 0.5)

    def test_differential_lr_values(self):
        config = self._load("resnet50_24gb_diff_lr.yaml")
        self.assertEqual(
            config["optimizer"]["differential_lr"],
            {"encoder_lr": 0.001, "decoder_lr": 0.01},
        )


if __name__ == "__main__":
    unittest.main()
