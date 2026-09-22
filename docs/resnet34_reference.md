# ResNet34 已完成实验基准

这份记录固定当前可用于答辩、网页演示和后续对照实验的 ResNet34 结果。

## 已确认结果

- 模型：ImageNet 预训练 ResNet34 编码器 + U-Net 3+ Decoder
- Full Validation 单尺度 mIoU：`0.4669311579969285`
- Full Validation mean Dice：`0.632850971221246`
- 最佳轮次：`epoch 16`
- LoveDA 官方隐藏 Test 单尺度 mIoU：`0.457311`
- Codabench submission ID：`936563`

## 可复现资产

- 代码 Git tag：`resnet34-val04669-test04573`
- 原实验目录：`runs/resnet34_pretrained_512_randomcrop/`
- 最佳权重：`runs/resnet34_pretrained_512_randomcrop/best_model.pth`
- 官网提交：`submissions/loveda_test_single_scale.zip`
- 本机独立归档：`/home/kxcc/unet3_archives/resnet34_val04669_test04573_20260921/`

权重和提交包未上传 GitHub，需要保留本机归档。

## 文件校验值

- `best_model.pth` SHA256：`b5f7ea0cfac8b209e23f41d0d1174c327acc0b8e7c9bd1dc2333a828ec443117`
- `loveda_test_single_scale.zip` SHA256：`6906617d42e458c9f64dd347018bc0ddee9ad1df77c7521a19dd18473648a4da`

后续 ResNet50 实验必须使用独立输出目录
`runs/resnet50_pretrained_512_randomcrop/`，不得覆盖上述文件。
