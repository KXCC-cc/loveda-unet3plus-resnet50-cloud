# LoveDA ResNet50 云端重跑说明

这个仓库是独立于原项目的云端实验快照，目标是重新运行本地RTX 4060 8GB上
因显存不足而失败的完整ResNet50 + U-Net 3+实验。

## 仓库包含什么

- ResNet50 ImageNet预训练编码器
- U-Net 3+ Full-scale Skip Connections
- U-Net 3+ Deep Supervision
- 512×512随机裁块
- CE + Dice
- SGD + poly学习率
- 15000个optimizer updates
- 16GB与24GB云GPU配置
- LoveDA数据集Release分卷下载脚本
- 本地OOM记录和后续小显存试验指标快照

旧checkpoint和历史训练输出没有放入代码仓库。它们不是从头云端重训所必需的，
而且单个文件超过GitHub普通文件100MB限制。

## 云端使用

    git clone https://github.com/KXCC-cc/loveda-unet3plus-resnet50-cloud.git
    cd loveda-unet3plus-resnet50-cloud
    bash scripts/cloud_prepare.sh

正式脚本会先执行 CUDA 与数据集预检。只有检测到 GPU，且 Train/Val 分别为
2522/1669 张并通过 image/mask 文件名核对后才会进入训练。

16GB GPU先运行：

    bash scripts/cloud_train_resnet50_16gb.sh

24GB及以上GPU可运行：

    bash scripts/cloud_train_resnet50_24gb.sh

这条命令是第一轮推荐实验，仍代表干净的 ResNet50 baseline。不要跳过它直接把
所有优化合并，否则无法判断收益来源。

若24GB配置仍OOM，改用16GB配置。不要通过减小裁块或cat_channels来悄悄改变
本次对照实验。两套云配置的effective batch都是16，线性学习率缩放后使用
lr=0.01。

## 输出位置

训练结果保存在：

    runs/cloud_resnet50_16gb/
    runs/cloud_resnet50_24gb/

包括best_model.pth、last_checkpoint.pth、metrics.csv、summary.json、
曲线、预测样本和混淆矩阵。

## 40GB及以上GPU

A100 40GB等更大显存GPU可减少串行梯度累积：

    bash scripts/cloud_train_resnet50_40gb.sh

三套配置都保持effective batch=16。由于batch=1 + accumulation=16会执行更多串行
前后向计算，16GB配置最省显存但最慢；优先选择能够容纳的最大物理batch。

验证间隔设置为8个数据轮次，约每1260次optimizer update执行一次完整验证，
避免effective batch增大后仍每2轮验证而明显拖慢训练。

无论本轮是否执行 Full Validation，每个完整 epoch 结束都会原子更新
`runs/<run>/last_checkpoint.pth`。非验证 epoch 只保存恢复点，不额外运行验证。
云实例中断后继续同一个 24GB baseline：

```bash
bash scripts/cloud_train_resnet50_24gb.sh \
  --resume runs/cloud_resnet50_24gb/last_checkpoint.pth
```

恢复会从下一个 epoch 开始，并恢复 optimizer update、poly scheduler、AMP scaler、
best mIoU 与随机状态。

## 受控消融

完成 baseline 后按顺序运行：

```bash
bash scripts/cloud_train_resnet50_24gb_diff_lr.sh
bash scripts/cloud_train_resnet50_24gb_classaware.sh
bash scripts/cloud_train_resnet50_24gb_multiscale.sh
bash scripts/cloud_train_resnet50_24gb_focal.sh
bash scripts/cloud_train_resnet50_24gb_warmup.sh
```

只有多尺度与类别感知单项都有效时，再运行：

```bash
bash scripts/cloud_train_resnet50_24gb_multiscale_classaware.sh
```

各项原理、类别影响和资源代价见 [CHANGELOG_MIOU.md](CHANGELOG_MIOU.md)。

## 显存预检

下面的命令模拟 ResNet50、cat64、512 crop、batch2、Deep Supervision、
CE+Dice、AMP、forward/backward/optimizer step，不会启动长训练：

```bash
python -m tools.benchmark_memory \
  --backbone resnet50 --batch-size 2 --image-size 512 \
  --cat-channels 64 --loss-profile ce_dice
```

如果 batch2 OOM，改用 `configs/cloud/resnet50_16gb.yaml`，保持 cat64 与 512 crop。
