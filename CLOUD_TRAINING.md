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

16GB GPU先运行：

    bash scripts/cloud_train_resnet50_16gb.sh

24GB及以上GPU可运行：

    bash scripts/cloud_train_resnet50_24gb.sh

若24GB配置仍OOM，改用16GB配置。不要通过减小裁块或cat_channels来悄悄改变
本次对照实验。两套云配置的effective batch都是16，线性学习率缩放后使用
lr=0.01。

## 输出位置

训练结果保存在：

    runs/cloud_resnet50_16gb/
    runs/cloud_resnet50_24gb/

包括best_model.pth、last_checkpoint.pth、metrics.csv、summary.json、
曲线、预测样本和混淆矩阵。
