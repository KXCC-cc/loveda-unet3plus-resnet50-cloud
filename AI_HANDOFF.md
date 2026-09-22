# LoveDA U-Net 3+ 接管提示词（旧 baseline 历史记录）

> 本文件记录 ResNet 重构前的 scratch baseline。当前入口、配置与命令以
> `README.md` 为准；旧训练入口已保存在 `train_legacy.py`。

请接管我的 PyTorch U-Net 3+ 遥感七分类项目。下面是交接信息，请先阅读实际源码和已保存结果，再决定下一步，以磁盘中的最新文件为准。你的目标是提高完整验证集的真实分割效果，同时控制实验耗时，并保持代码适合学习。

## 1. 位置、环境与工作边界

- 项目已经从 Windows D 盘迁移到 WSL Ubuntu：`/home/kxcc/unet3`。
- Windows 对应路径：`\\wsl.localhost\Ubuntu\home\kxcc\unet3`。
- 已有 WSL Python：`/home/kxcc/venvs/unet3/bin/python`。后续命令明确使用这个解释器；不要擅自改用项目 `.venv`、Windows Python 或另建环境。
- 原实验运行日志记录 RTX 4060 Laptop GPU、PyTorch 2.9.1+cu126、AMP；接管时仍需以实际当前环境为准，不能凭安装目录判定 CUDA 已正常工作。
- 只在项目目录内修改文件。禁止修改、移动或删除 dataset 原始数据，不覆盖原实验和最佳权重。
- 接管的第一步是只读审阅并说明方案；不要因为看到本文的命令就立即启动训练、安装、下载或重跑已有诊断。是否运行以我的后续明确要求为准。
- 最近的微调代码改动只经过静态阅读，没有运行 Python、forward、测试或训练。已存在的训练/诊断报告是另行实际执行产生的结果，必须区分两者。
- 中文沟通、教学型注释、数据流清晰；不引入 Lightning、Hydra、复杂 registry 或大型实验框架。

## 2. 数据与网络

LoveDA 原图为 RGB，Train/Val 分 Rural、Urban，均有 images_png 与 masks_png；Test 只有 images_png。已检查的数据规模为 Train 2522 张、Val 1669 张，原图通常为 1024×1024；仍应按当前目录核对。

原始 mask 的 0 是 no-data，映射为 255；1～7 映射为训练类 0～6。类别顺序固定为 background、building、road、water、barren、forest、agricultural，训练类别 0 是有效背景类。常量和映射集中在 utils/constants.py。

Dataset 严格按文件名配对，Train 原生随机 crop，image/mask 同步翻转和直角旋转；ImageNet mean/std 归一化；mask 只有 resize 对照策略才使用 nearest。Val/Test 默认保留全图，确定性滑窗，先融合 logits 再 argmax。不要把整图缩小后的指标与原分辨率指标混为一谈。

网络版本 `unet3plus_loveda_v2`：五层 encoder 64/128/256/512/1024；每个 decoder 显式融合五路全尺度特征，每路投影为 64 通道，concat/fusion 为 320 通道。D1 为 main，D2/D3/D4/E5 为四个 aux，均返回七通道原始 logits，辅助权重为 0.5/0.25/0.125/0.0625。没有预训练 encoder，没有 CGM。

默认 CE+Dice，Focal 可选但默认关闭，ignore=255。`--class-weights auto` 读取项目 class_stats.json 的 Train 真实频率，计算平方根倒数并归一化到均值 1；权重作用于 CE/Focal，不作用于 Dice。不要假造频率或随意加强类别权重。

优化器 AdamW，scheduler CosineAnnealingLR，CUDA AMP、梯度裁剪、完整 Val、mIoU 最佳模型与 resume 均已实现。保持网络、Dataset、Loss、Deep Supervision、optimizer/scheduler 算法；大改之前先提出依据。

## 3. 必读文件

- README.md，尤其第 19、20 节。
- models/blocks.py、models/unet3plus.py。
- utils/constants.py、dataset.py、transforms.py、losses.py、metrics.py。
- utils/inference.py、device.py、checkpoint.py、experiment.py。
- train.py、predict.py、diagnose.py、test_model.py、requirements.txt。
- class_stats.json、compute_class_stats.py。
- checkpoints/auto_weights_40e/{config.json,metrics.csv,summary.json}。
- checkpoints/diagnostics_20260919_142359_262161/{config.json,comparison.csv,report.json,samples.json}。

旧 ANTIGRAVITY_RUN_PROMPT.md 与 Windows 环境辅助脚本属于历史资料，不能覆盖这里的 WSL 环境与最新结果。不要把 test_model.py 的存在说成已经通过测试。

## 4. 已完成的原实验

实验目录：`/home/kxcc/unet3/checkpoints/auto_weights_40e`。

实际参数：40 epochs、batch 2、crop 256×256、samples_per_image=1、eval_stride=256×256、workers=4、lr=3e-4、weight_decay=1e-4、AMP、seed=42、auto 类别权重、CE+Dice、四辅助头权重不变。实际实验参数与 train.py 的默认参数不完全相同，特别是每图取样次数。

用户报告整次训练约 16 小时。最佳 mIoU 模型是 `best_model.pth`，来自 epoch 30：

- 完整 Val mIoU：0.3456975087，即 34.57%。
- 该轮 Dice：0.5058852649。
- 该轮 Val Main Loss：2.1410533365。
- 该轮逐类 IoU：background 46.21%、building 40.56%、road 36.58%、water 32.69%、barren 14.14%、forest 33.50%、agricultural 38.31%。

历史最高 Dice 在 epoch 34，最低 Val Loss 在 epoch 36；不能说三项最佳来自同一轮。根目录 confusion_matrix 对应最后完成的 epoch，不一定对应最佳权重。Train Loss 包含深监督，Val Main Loss 只有主输出，数值不可直接当成同一个损失比较。

## 5. 已完成的诊断：不要重复从头跑

`checkpoints/diagnostics_20260919_142359_262161/report.json` 的 status 为 completed，使用上述 epoch 30 权重、固定 Val 100 张（Rural/Urban 各 50）。

| 设置 | Overall mIoU | Rural mIoU | Urban mIoU | 评估耗时 |
| --- | --- | --- | --- | --- |
| baseline | 32.1685% | 23.8502% | 36.7313% | 53.15 秒 |
| overlap | 32.4551% | 24.0972% | 36.9888% | 153.80 秒 |
| bn | 32.0242% | 22.5664% | 37.3824% | 51.38 秒 |
| bn_overlap | 32.3087% | 22.8135% | 37.6022% | 154.03 秒 |

baseline stride=256，overlap stride=128，tile 均为 256，窗口 batch=2、GPU FP32 融合、AMP。BN 只用 Train 的 256 个裁块、128 批校准，未反传；校准统计只留在诊断进程内，没有覆盖原权重。

可作出的判断：重叠仅增加约 0.29 个百分点，耗时约 2.89 倍；这次 BN 校准总体略降，农业 IoU 34.18%→27.57%。当前结果不支持把 BN 校准作为主要提分措施。乡村表现较弱，但不能由此断言乡村图片数量不足，或证明采样覆盖不足就是唯一原因。

100 张均衡子集分数只能在四组之间比较，不能直接与完整 Val 的 34.57% 比较。时间含读取与初始化，不是严格性能 benchmark，也不能用来断言原 16 小时训练中验证占比。

## 6. 最近已写入、尚未运行的微调改动

train.py 和 utils/checkpoint.py 已增加：

1. `--init-checkpoint`：严格载入全部模型参数及原 BN buffers，仅作为新实验起点；重新创建 optimizer/scheduler/scaler，新轮数从 1 开始。
2. 与 `--resume` 互斥。resume 仍是同一实验的完整恢复，不能随意更改 epochs/LR；它保留微调来源信息。
3. 新实验目录必须不存在或为空（允许 .gitkeep），防止覆盖原实验；resume 使用原实验目录。
4. `--eval-tile-batch-size`、`--eval-accumulate-on-device`：把已有批量滑窗/GPU 融合接到训练验证，旧默认仍为 batch1/CPU。
5. 新微调先完整验证起点，写 initial_validation.json，不伪造 epoch 0 的训练日志。
6. summary.json 新增 initial_validation_miou、best_miou_gain_over_initial、improved_over_initial；新实验最佳未必超过原模型。
7. timings.json 记录每轮 Train/Val/保存绘图时间和峰值显存；resume 按恢复 epoch 对齐记录。

本次没有改模型、Dataset、Loss、DS 或 AdamW/cosine 算法。请静态检查最新接口后，再给我执行建议；若磁盘已经有后续微调结果，应先读取新结果，而不是沿用“尚未运行”的交接状态。

## 7. 当前拟议实验与命令

先保留 256 patch / 256 stride，避免同时改太多因素。用 epoch 30 最佳权重、新 lr=5e-5、每图每轮 2 次裁块、12 轮微调；其余损失与增强参数不变。lr 接近原最佳阶段；额外裁块增加取样覆盖。这是待验证假设，不保证提升，仍有真实训练成本。

下面只是待用户执行的命令，不代表你现在有权启动它：

```bash
cd /home/kxcc/unet3
/home/kxcc/venvs/unet3/bin/python train.py \
  --init-checkpoint checkpoints/auto_weights_40e/best_model.pth \
  --checkpoint-dir checkpoints/finetune_best30_12e \
  --device cuda:0 --amp \
  --epochs 12 --batch-size 2 \
  --data-strategy crop --crop-size 256 256 \
  --samples-per-image 2 \
  --lr 5e-5 --weight-decay 1e-4 \
  --class-weights auto \
  --eval-stride 256 256 \
  --eval-tile-batch-size 2 --eval-accumulate-on-device \
  --num-workers 4 --seed 42
```

若目录已存在，先确认是否已有训练结果，不覆盖。若中断后已有 last_checkpoint.pth，在同一目录用 --resume 替换 --init-checkpoint，其他训练参数保持一致，保留 metrics.csv 和 initial_validation.json。若初始验证尚未完成就中断，没有 checkpoint，应换新目录重新开始。

之后判断结果时，比较同一完整 Val 协议下的 initial_validation 与新 summary，检查逐类 IoU、农业/裸地混淆及耗时。若微调没有实际改善，不要无止境叠加训练轮数，应先分析 Train/Val 差距、场景泛化、训练覆盖和更大视野等假设。384/512 裁块增加上下文但显存/计算量更高，应作为后续独立实验；如分离训练 crop 与评估 tile，需要同步检查 predict、diagnose 和 checkpoint 元信息。

## 8. 展示与最终交付

将来需要网页/UI 展示，现在不开发。demo_assets/Rural、demo_assets/Urban 各保留 20 张 Test 副本，manifest 记录来源；Test 无真值，只作展示推理，不用于训练、调参或计算真实 mIoU。

接管后的第一次回复，请说明：你实际读到了哪些最新文件；哪些实验已经完成；当前改动是否与源码一致；建议下一步做什么及依据。不要声称你没有亲自执行或没有结果证据的测试、训练、CUDA 检查已经成功。
