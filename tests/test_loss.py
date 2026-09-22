import unittest

import torch

from utils.losses import (
    CombinedLoss,
    CrossEntropyLoss,
    FocalLoss,
    deep_supervision_loss,
)


class LossTests(unittest.TestCase):
    def test_focal_gamma_zero_matches_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 6, 5)
        target = torch.randint(0, 7, (2, 6, 5))
        target[0, 0, 0] = 255
        focal = FocalLoss(gamma=0.0)(logits, target)
        ce = CrossEntropyLoss()(logits, target)
        self.assertTrue(torch.allclose(focal, ce, atol=1e-6, rtol=1e-6))

    def test_focal_ignore_index_matches_valid_subset(self):
        logits = torch.tensor([[[[2.0, -1.0]], [[-1.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[0, 255]]])
        actual = FocalLoss(num_classes=2, gamma=2.0)(logits, target)
        expected = FocalLoss(num_classes=2, gamma=2.0)(
            logits[:, :, :, :1], torch.tensor([[[0]]])
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))

    def test_focal_all_ignore_is_finite_zero(self):
        logits = torch.randn(1, 7, 4, 4, requires_grad=True)
        target = torch.full((1, 4, 4), 255, dtype=torch.long)
        loss = FocalLoss()(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_backward_is_finite(self):
        logits = torch.randn(2, 7, 5, 5, requires_grad=True)
        target = torch.randint(0, 7, (2, 5, 5))
        loss = FocalLoss(gamma=2.0)(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighted_focal_backward_is_finite(self):
        logits = torch.randn(1, 7, 5, 5, requires_grad=True)
        target = torch.randint(0, 7, (1, 5, 5))
        loss = FocalLoss(class_weights=[1, 2, 3, 4, 5, 6, 7])(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_combined_loss_skips_zero_weight_branches(self):
        class FailIfCalled(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                raise AssertionError("zero-weight loss branch was evaluated")

        criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.0, focal_weight=0.0)
        criterion.dice = FailIfCalled()
        criterion.focal = FailIfCalled()
        logits = torch.randn(1, 7, 4, 4, requires_grad=True)
        target = torch.randint(0, 7, (1, 4, 4))
        loss = criterion(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
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
