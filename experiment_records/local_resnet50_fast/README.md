# 本地缩减版ResNet50最终结果

这是RTX 4060 Laptop 8GB为了能够运行而采用的缩减对照实验，已经正常完成。
它不是云端计划运行的完整ResNet50配置。

## 配置区别

- backbone：ImageNet预训练ResNet50
- crop：512×512
- cat_channels：48（完整配置为64）
- physical batch：1
- accumulation：2
- effective batch：2
- max optimizer steps：8000
- AMP：开启
- 验证：完整LoveDA Validation，共1669张

## 最终结果

| 指标 | 数值 |
|---|---:|
| best/full-Val mIoU | 0.456944 |
| best/full-Val mean Dice | 0.620103 |
| best mIoU epoch | 7 |
| best mIoU epoch val loss | 1.753550 |
| best val loss | 1.729073（epoch 2） |

### 最佳epoch每类IoU

| 类别 | IoU |
|---|---:|
| background | 0.516369 |
| building | 0.545343 |
| road | 0.434877 |
| water | 0.572983 |
| barren | 0.245403 |
| forest | 0.420145 |
| agricultural | 0.463491 |

## 与ResNet34基线比较

旧ResNet34完整Validation最佳mIoU约为0.4669。本实验为0.4569，低约0.0100，
因此没有超过现有主模型。结果说明在8GB显存约束下，仅更换ResNet50，同时降低
decoder通道、effective batch和训练预算，不能保证获得更高mIoU。

云端实验将恢复cat_channels=64、effective batch=16和15000次optimizer update，
用于检验完整ResNet50配置是否能够发挥更强编码器的作用。

## 文件用途

- config.json：实际运行参数与环境
- metrics.csv：epoch 2/4/6/7完整验证指标
- summary.json：最佳轮次和每类IoU
- confusion_matrix.csv/png：最终完整验证混淆矩阵
- curves/：损失、mIoU、Dice、学习率和每类曲线
- predictions/best/：固定验证样本的RGB、GT、预测和并排对比图

best_model.pth与last_checkpoint.pth约238MB，保留在本机原运行目录，没有写入
Git历史。
