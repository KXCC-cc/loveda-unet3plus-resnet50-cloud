import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.constants import IGNORE_INDEX
from utils.dataset import LoveDADataset
from utils.transforms import transform_image_and_mask


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

    def test_multiscale_resize_preserves_discrete_mask_values(self):
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        raw_mask = np.ones((8, 8), dtype=np.uint8)
        raw_mask[:, 4:] = 7
        _, transformed = transform_image_and_mask(
            image,
            raw_mask,
            image_size=(8, 8),
            spatial_mode="crop",
            train_scales=(0.5,),
        )
        self.assertTrue(set(np.unique(transformed)).issubset({0, 1, 7}))

    def test_class_aware_failure_falls_back_to_random_crop(self):
        image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        raw_mask = np.ones((16, 16), dtype=np.uint8)
        transformed_image, transformed_mask = transform_image_and_mask(
            image,
            raw_mask,
            image_size=(8, 8),
            spatial_mode="crop",
            class_aware_crop_probability=1.0,
            class_aware_raw_classes=(5,),
            min_target_pixels=64,
            max_crop_attempts=1,
        )
        self.assertEqual(transformed_image.size, (8, 8))
        self.assertEqual(transformed_mask.shape, (8, 8))
        self.assertTrue(np.all(transformed_mask == 1))


if __name__ == "__main__":
    unittest.main()
