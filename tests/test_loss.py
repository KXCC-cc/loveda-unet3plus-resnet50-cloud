import unittest

import torch

from utils.losses import CombinedLoss, CrossEntropyLoss, deep_supervision_loss


class LossTests(unittest.TestCase):
    def test_ignore_pixels_are_supported(self):
        logits = torch.randn(1, 7, 8, 8, requires_grad=True)
        target = torch.randint(0, 7, (1, 8, 8))
        target[:, :2] = 255
        loss = CombinedLoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_deep_supervision_heads(self):
        target = torch.randint(0, 7, (1, 8, 8))
        outputs = {
            "main": torch.randn(1, 7, 8, 8, requires_grad=True),
            "aux": [torch.randn(1, 7, 8, 8, requires_grad=True) for _ in range(4)],
        }
        loss = deep_supervision_loss(outputs, target, CrossEntropyLoss())
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
