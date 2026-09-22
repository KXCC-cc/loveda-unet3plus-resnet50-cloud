import unittest
from pathlib import Path

from utils.constants import CLASS_NAMES, RESNET34_REFERENCE_MIOU
from utils.experiment import build_summary


class ExperimentSummaryTests(unittest.TestCase):
    def test_summary_reports_delta_against_resnet34(self):
        row = {
            "epoch": 1,
            "train_loss": 1.0,
            "val_main_loss": 1.0,
            "mean_dice": 0.6,
            "miou": 0.5,
            "learning_rate": 0.01,
        }
        row.update({f"{name}_iou": 0.5 for name in CLASS_NAMES})
        row.update({f"{name}_dice": 0.6 for name in CLASS_NAMES})
        summary = build_summary(
            [row],
            Path("best_model.pth"),
            "complete",
            baseline_reference_miou=RESNET34_REFERENCE_MIOU,
        )
        self.assertEqual(
            summary["baseline_reference_miou"], RESNET34_REFERENCE_MIOU
        )
        self.assertAlmostEqual(
            summary["delta_vs_resnet34"], 0.5 - RESNET34_REFERENCE_MIOU
        )


if __name__ == "__main__":
    unittest.main()
