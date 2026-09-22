import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.constants import IGNORE_INDEX
from utils.dataset import LoveDADataset


class DatasetTests(unittest.TestCase):
    def test_label_mapping_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for region in ("Rural", "Urban"):
                image_dir = root / "Val" / region / "images_png"
                mask_dir = root / "Val" / region / "masks_png"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "a.png")
                mask = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
                Image.fromarray(mask).save(mask_dir / "a.png")
            dataset = LoveDADataset(root, split="Val", spatial_mode="full")
            image, mask = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(mask.dtype, torch.long)
            self.assertEqual(int(mask[0, 0]), IGNORE_INDEX)
            self.assertEqual(mask[0, 1:].tolist(), list(range(7)))


if __name__ == "__main__":
    unittest.main()
