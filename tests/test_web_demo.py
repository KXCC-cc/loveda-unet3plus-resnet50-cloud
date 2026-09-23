import unittest

import numpy as np
from PIL import Image

from utils.experiment import CLASS_COLORS
from web_demo.app import build_model_visual
from web_demo.model_registry import MODEL_SPECS


class WebDemoTests(unittest.TestCase):
    def test_registry_order_and_official_scores(self):
        self.assertEqual(
            [spec.id for spec in MODEL_SPECS],
            [
                "resnet34_baseline",
                "resnet50_cat48",
                "resnet34_diff_lr",
                "resnet50_cloud_final",
            ],
        )
        self.assertEqual(
            [spec.test_miou for spec in MODEL_SPECS],
            [0.457311, 0.446118, 0.454974, 0.469813],
        )

    def test_registered_model_identities(self):
        expected = [
            ("resnet34", 64, False),
            ("resnet50", 48, False),
            ("resnet34", 64, True),
            ("resnet50", 64, False),
        ]
        actual = [
            (
                spec.expected_backbone,
                spec.expected_cat_channels,
                spec.requires_differential_lr,
            )
            for spec in MODEL_SPECS
        ]
        self.assertEqual(actual, expected)

    def test_visual_uses_shared_loveda_palette(self):
        source = Image.new("RGB", (7, 1), color=(120, 130, 140))
        prediction = np.arange(7, dtype=np.uint8).reshape(1, 7)
        result = build_model_visual(
            source,
            prediction,
            MODEL_SPECS[0],
            elapsed=1.25,
            load_elapsed=0.5,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["mask"].startswith("data:image/png;base64,"))
        self.assertTrue(result["overlay"].startswith("data:image/png;base64,"))
        self.assertEqual(CLASS_COLORS.shape, (7, 3))


if __name__ == "__main__":
    unittest.main()
