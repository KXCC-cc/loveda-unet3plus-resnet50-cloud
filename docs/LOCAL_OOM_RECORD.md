# RTX 4060 8GB 本地失败记录

## 失败配置

- 模型：ResNet50 + U-Net 3+
- 输入裁块：512×512
- U-Net 3+ cat_channels=64
- Deep Supervision：开启
- AMP：开启
- physical batch：1
- accumulation：4
- max optimizer steps：15000

## 错误位置

错误发生在训练反向传播：

    scaler.scale(loss / accumulation).backward()
    torch.OutOfMemoryError: CUDA out of memory
    Tried to allocate 2.00 GiB
    GPU total capacity: 8.00 GiB

这是8GB显存容量不足，不是模型代码或依赖包缺失。后续本地试验曾将
cat_channels降为48、累积步数降为2、总步数降为8000，使训练能够运行，
但它已经不是这里保存的完整ResNet50对照配置。

云端重跑应优先保留cat_channels=64，并根据显存选择
configs/cloud/resnet50_16gb.yaml或configs/cloud/resnet50_24gb.yaml。
