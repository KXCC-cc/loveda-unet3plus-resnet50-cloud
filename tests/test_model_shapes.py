"""模型结构测试默认不下载预训练权重。"""

import unittest

import torch

from models import UNet3Plus, UNet3PlusResNet
from models.backbones import FrozenBatchNorm2d


class ModelShapeTests(unittest.TestCase):
    def test_resnet34_outputs(self):
        model = UNet3PlusResNet(backbone="resnet34", pretrained=False).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64), return_aux=True)
        self.assertEqual(tuple(outputs["main"].shape), (1, 7, 64, 64))
        self.assertEqual([tuple(x.shape) for x in outputs["aux"]], [(1, 7, 64, 64)] * 4)

    def test_resnet50_main_only(self):
        model = UNet3PlusResNet(backbone="resnet50", pretrained=False).eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 64, 64), return_aux=False)
        self.assertEqual(tuple(outputs["main"].shape), (1, 7, 64, 64))
        self.assertEqual(outputs["aux"], [])

    def test_encoder_bn_is_frozen(self):
        model = UNet3PlusResNet(backbone="resnet34", pretrained=False, freeze_encoder_bn=True)
        model.train()
        self.assertTrue(any(isinstance(module, FrozenBatchNorm2d) for module in model.encoder.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.BatchNorm2d) for module in model.encoder.modules()))

    def test_legacy_model_still_builds(self):
        model = UNet3Plus().eval()
        with torch.no_grad():
            outputs = model(torch.randn(1, 3, 32, 32), return_aux=False)
        self.assertEqual(tuple(outputs["main"].shape), (1, 7, 32, 32))


if __name__ == "__main__":
    unittest.main()
