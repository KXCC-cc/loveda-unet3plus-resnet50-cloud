# LoveDA U-Net 3+ ResNet50 云端重训快照

本仓库用于在云GPU上重新运行本地RTX 4060 8GB因显存不足而失败的完整ResNet50实验。
云端环境、数据下载和启动命令见 [CLOUD_TRAINING.md](CLOUD_TRAINING.md)，本地失败记录见
[docs/LOCAL_OOM_RECORD.md](docs/LOCAL_OOM_RECORD.md)。

---

# LoveDA U-Net 3+：Scratch Baseline 与 ImageNet ResNet Encoder

本项目用于 LoveDA 七分类遥感语义分割。仓库同时保留旧 scratch U-Net 3+
baseline 和新增的 ResNet34/50 U-Net 3+。改进目标是在完整 Validation Set 上取得
`mIoU >= 0.45`，它只是实验目标，最终结果必须来自完整验证集真实累计的混淆矩阵。

参考资料：

- LoveDA 官方仓库：https://github.com/Junjue-Wang/LoveDA
- LoveDA 论文：https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf
- U-Net 3+ 论文：https://arxiv.org/abs/2004.08790

## 三套实验定位

### 1. Baseline-Scratch-UNet3Plus

旧模型完整保存在 `models/unet3plus.py`，旧训练与推理入口分别保存在
`train_legacy.py` 和 `predict_legacy.py`。旧 checkpoint 不会被新 run 覆盖。

旧实验约为：scratch encoder、256 crop、CE + Dice、auto class weights、AdamW、
Cosine，历史最佳完整 Val mIoU 约 0.3457。`configs/baseline_scratch.yaml` 记录了
对应对照配置，新旧入口都保留。

### 2. LoveDA official reproduction profile

`configs/loveda_official_resnet50.yaml` 记录官方论文/仓库的核心训练协议：

- ImageNet 预训练 ResNet50
- 512×512 random crop
- SGD，momentum=0.9，weight decay=1e-4
- base lr=0.01，poly power=0.9
- 15000 optimizer updates
- physical batch=16
- 随机镜像和 90° 旋转
- CE，禁止 auto class weights
- 多尺度集合 `{0.5, 0.75, 1.0, 1.25, 1.5, 1.75}`

官方仓库的 `configs/base/loveda.py` 基础数据管线是 512 random crop 加
horizontal/vertical/rotate90 三选一；论文另行报告多尺度训练/测试实验。本项目的
official YAML 明确启用上述六尺度训练，不把它悄悄混入其他配置。官方 TTA 代码对
各尺度 logits 恢复到原尺寸后求平均，本项目也按此顺序实现。

这个 profile 的 `batch=16` 不适合 RTX 4060 Laptop 8GB，脚本只用于协议对照或
更大显存设备，不能把小 batch 适配写成“严格官方 batch 16”。

### 3. RTX 4060 performance profile

`configs/loveda_4060_resnet34.yaml` 是第一轮推荐实验：ResNet34 ImageNet 权重、
512 crop、CE + Dice、单尺度训练、physical batch=1、累积 4 次、AMP、冻结 Encoder
BN、Decoder GroupNorm、SGD/poly。有效 batch=4 时启用线性 LR 缩放，实际初始学习率
为 `0.01 × 4 / 16 = 0.0025`。

第一轮有意关闭 class-aware crop、Focal、多尺度测试和 color jitter，先隔离
“预训练 Encoder + 512 crop”的收益。之后按顺序运行：

1. `loveda_4060_resnet34.yaml`
2. `loveda_4060_resnet50.yaml`（只更换 backbone，作为公平对照）
3. `resnet_pretrained_512_multiscale.yaml`
4. `resnet_pretrained_512_classaware.yaml`
5. 对最佳 checkpoint 做六尺度最终评估

已完成的 ResNet34 基准不会被后续实验覆盖。验证集、官方 Test 成绩、权重哈希、
Git 标签和本机归档位置记录在 `docs/resnet34_reference.md`。

## 项目结构

```text
configs/                 official、4060、baseline 与消融 YAML
engine/                  trainer 与完整 Val evaluator
models/
  unet3plus.py           原 scratch baseline（保留）
  unet3plus_resnet.py    新 ResNet U-Net 3+
  backbones/resnet.py    segmentation stem 与 FrozenBatchNorm2d
scripts/                 可直接运行的训练/评估脚本
docs/                    已确认实验结果与本地归档索引
tests/                   shape、Dataset、Loss 测试
tools/                   checkpoint 评估、显存测试、统计和 run 对比
utils/                   数据、增强、损失、指标、checkpoint、实验记录
train.py                 新配置化入口
train_legacy.py          原 baseline 入口
predict.py               新 checkpoint 推理
predict_legacy.py        原 checkpoint 推理
dataset/                 LoveDA 原始数据（Git 忽略）
runs/                    新实验输出（Git 忽略）
```

## LoveDA 标签

原始 mask 的类别定义集中在 `utils/constants.py`：

| 原始值 | 训练索引 | 类别 |
|---:|---:|---|
| 0 | 255 | no-data / ignore |
| 1 | 0 | background |
| 2 | 1 | building |
| 3 | 2 | road |
| 4 | 3 | water |
| 5 | 4 | barren |
| 6 | 5 | forest |
| 7 | 6 | agricultural |

`ignore_index=255` 不参与 CE、Dice、混淆矩阵、IoU 或验证均值。训练、验证、推理
不得在其他文件重新定义标签顺序。

数据目录：

```text
dataset/{Train,Val}/{Rural,Urban}/{images_png,masks_png}
dataset/Test/{Rural,Urban}/images_png
```

Dataset 严格按同名 PNG 配对，发现缺失或多余 mask 会直接报错。

## 一个样本从 PNG 到 Loss

```text
RGB PNG + 原始索引 mask PNG
  -> 同比例多尺度 resize（image bilinear，mask nearest）
  -> 尺寸不足时 padding（mask 用原始 ignore=0）
  -> 512×512 random crop / 可选 class-aware crop
  -> 同步水平翻转、垂直翻转、90°旋转
  -> image 转 float32、[0,1]、ImageNet mean/std normalization
  -> mask: 0->255，1..7->0..6，torch.long
  -> DataLoader: [B,3,512,512] + [B,512,512]
  -> non_blocking 传入 CUDA
  -> ResNet Encoder + U-Net 3+ full-scale Decoder
  -> main logits + 4 个辅助 logits（均为 [B,7,512,512]）
  -> CE 或 CE+Dice Deep Supervision Loss
  -> AMP scaled backward
  -> 累积指定 micro batches 后 optimizer update
  -> poly scheduler 按 optimizer update 更新
```

任何几何操作都同步作用于 image 与 mask。mask 只用 nearest，不允许 bilinear 产生
不存在的类别。Val/Test 不做随机增强。

## ResNet U-Net 3+ 结构

默认 `standard_resnet_stem=false`。先加载 torchvision 官方 ImageNet 权重，再把
ResNet `conv1.stride` 从 2 改为 1。7×7 卷积权重 shape 没变，所以预训练权重仍可
完整加载，同时避免分类模型在 stem 中过早降到 H/4。

512 输入时的 Encoder：

| 特征 | 空间尺寸 | ResNet34 通道 | ResNet50 通道 |
|---|---:|---:|---:|
| E1 | 512×512 | 64 | 64 |
| E2 | 256×256 | 64 | 256 |
| E3 | 128×128 | 128 | 512 |
| E4 | 64×64 | 256 | 1024 |
| E5 | 32×32 | 512 | 2048 |

`standard_resnet_stem=true` 保留分类 ResNet 的 H/2～H/32 五尺度，供消融；最终
main logits 仍插值回输入尺寸。

Decoder 没有改成普通 U-Net。每条支路先对齐空间尺寸，再通过
`Conv + GroupNorm + ReLU` 投影到 64 通道，五路 concat 为 320 通道：

```text
D4: E1 + E2 + E3 + E4 + E5 -> [B,320,H/8,W/8]
D3: E1 + E2 + E3 + D4 + E5 -> [B,320,H/4,W/4]
D2: E1 + E2 + D3 + D4 + E5 -> [B,320,H/2,W/2]
D1: E1 + D2 + D3 + D4 + E5 -> [B,320,H,W]
```

代码在 `models/unet3plus_resnet.py` 中逐支路显式书写。训练时返回：

```python
{"main": d1_logits, "aux": [d2_logits, d3_logits, d4_logits, e5_logits]}
```

四个辅助 logits 在损失前上采样到 GT 尺寸。验证/推理调用
`return_aux=False`，省去辅助分类头和四套全分辨率 logits。

Encoder 的 BatchNorm 已转换为 FrozenBatchNorm2d，`model.train()` 不会恢复其
running statistics 更新；Decoder 使用 GroupNorm，避免 batch=1 时普通 BN 不稳定。

## Loss profiles

- official：`CrossEntropyLoss(ignore_index=255)`，不使用类别权重。
- performance：默认 `CE + Dice`，深监督权重为
  `[0.5, 0.25, 0.125, 0.0625]`。
- Focal 默认关闭。MS-SSIM 和医疗图像 CGM 没有加入：前者没有自然、稳定的七类
  LoveDA 定义；后者原本用于二分类目标存在性门控，会抑制正常存在的地表类别。
- `class_weights: auto` 作为消融保留，从真实 `class_stats.json` 计算
  `1/sqrt(frequency)` 并归一化到均值 1；官方 profile 不启用。

## Class-aware crop

这是 performance 选项，不属于 official 默认值。启用时一部分 crop 围绕 building、
road、water、barren 候选像素采样，并要求目标像素数达到阈值；有限次尝试失败后回退
普通 random crop。默认概率为 0，建议在完成随机裁块实验后单独做消融。

## 训练、CUDA 与 AMP

安装环境：

```bash
cd /home/kxcc/unet3
source ~/venvs/unet3/bin/activate
pip install -r requirements.txt
```

第一轮推荐命令：

```bash
cd /home/kxcc/unet3
bash scripts/train_4060_resnet34.sh
```

RTX 4060 上的 ResNet50 公平对照命令：

```bash
cd /home/kxcc/unet3
bash scripts/train_4060_resnet50.sh
```

该配置沿用 ResNet34 的 512 crop、CE + Dice、SGD/poly、batch=1、累积 4 次和
单尺度训练，只把 ImageNet Encoder 换成 ResNet50。它会写入独立目录
`runs/resnet50_pretrained_512_randomcrop/`，并从 ImageNet 权重开始新实验。

配置覆盖示例：

```bash
python train.py --config configs/loveda_4060_resnet34.yaml \
  --output-dir runs/try_batch2 \
  --set data.loader.batch_size=2 \
  --set training.accumulation_steps=2
```

显式请求 `cuda:0` 而 CUDA 不可用时会报错，不会静默切到 CPU。启动信息会显示
CUDA、GPU、AMP、physical batch、accumulation 和 effective batch。AMP 使用
autocast + GradScaler；梯度只在完整 accumulation 后裁剪和更新。AMP overflow 会
统计并每累计 5 次发出警告，发生 overflow 的 update 不推进 poly scheduler。

正式 ResNet50/batch16 对照命令：

```bash
bash scripts/train_official_resnet50.sh
```

不要在 8GB 机器直接运行这条命令；它是协议配置，不是 4060 配置。

## 验证与最终多尺度评估

普通训练验证固定为 scale=1.0、tile=512、stride=512，并通过
`patch_batch_size` 一次前向多个 tile。所有图片的混淆矩阵先累计，再计算 7-class
mIoU 和 mean Dice；没有使用逐图 IoU 平均。只有完整 Val 结果能更新 best model。

单尺度复评：

```bash
bash scripts/eval_single_scale.sh runs/resnet34_pretrained_512_randomcrop/best_model.pth
```

最终六尺度复评：

```bash
bash scripts/eval_multiscale.sh runs/resnet34_pretrained_512_randomcrop/best_model.pth
```

多尺度流程是：每个尺度滑窗 logits -> logits 插值回原图 -> 平均 logits -> argmax。
可显式追加 `--horizontal-flip`；官方仓库 TTA 模块支持镜像，但默认脚本只使用论文
列出的六尺度，分别报告 single-scale 与 multi-scale 结果。

## 实验输出与 checkpoint

每个 run 写入自己的目录：

```text
runs/<run_name>/
  config.json
  metrics.csv
  summary.json
  best_model.pth
  last_checkpoint.pth
  curves/*.png
  predictions/best/*
  predictions/epoch_XXXX/*
  confusion_matrix.csv
  confusion_matrix.png
```

为避免误删已有成果，新训练遇到非空输出目录会立即报错。断点续训时，
`--resume` checkpoint 必须位于当前 `run.output_dir` 中；如需开始另一组实验，必须
指定新的 run 目录。

`config.json` 记录 Git commit、PyTorch/CUDA/GPU、seed、实际 weights enum、crop、scale、
optimizer、scheduler、physical/effective batch 和解析后的 LR。checkpoint 保存 model、
optimizer、scheduler、GradScaler、epoch、micro step、optimizer step、best mIoU、RNG 和
完整配置。checkpoint 只在一个 optimizer update 完成后保存，不保存半次梯度累积。

恢复命令：

```bash
python train.py --config configs/loveda_4060_resnet34.yaml \
  --resume runs/resnet34_pretrained_512_randomcrop/last_checkpoint.pth
```

## 单图推理

```bash
python predict.py dataset/Test/Rural/images_png/123.png \
  runs/resnet34_pretrained_512_randomcrop/best_model.pth \
  --output predictions/123.png \
  --color-output predictions/123_color.png
```

输出 `prediction.png` 使用 LoveDA 提交值 1～7；彩色图使用项目固定颜色表。旧
checkpoint 请使用 `predict_legacy.py`。

## 测试与显存基准

```bash
python -m unittest discover -s tests -v
python -m tools.benchmark_memory --output memory_benchmark.json
```

`memory_benchmark.json` 保存的是重构初期在本机 RTX 4060 Laptop 8GB、PyTorch
2.9.1、AMP、512 输入下，五头 CE 反向和一次 SGD update 的实测：

| Backbone | batch | 结果 | peak allocated |
|---|---:|---|---:|
| ResNet34 | 1 | OK | 3.94 GiB |
| ResNet34 | 2 | 完成，但显存余量很小 | 7.70 GiB |
| ResNet50 | 1 | 完成，但余量较小 | 6.88 GiB |
| ResNet50 | 2 | OOM | — |

当前显存工具已改为与 performance profile 相同的五头 CE + Dice。ResNet50、batch=1
复核通过，peak allocated 为 6.88 GiB，peak reserved 为 8.41 GiB；因此正式配置仍固定
physical batch=1，并通过梯度累积得到 effective batch=4。

WSL 的 reserved/共享内存统计可能高于显卡物理容量，因此 batch2“完成”不代表适合
长训练；显存碎片、DataLoader 和验证缓存都会降低余量。第一轮固定推荐 ResNet34、
batch1、accum4。若要试 batch2，应先重新运行显存工具，再做几步短试验。

## 消融顺序与 0.45 目标

建议每次只改变一个变量：

| 实验 | 主要变化 |
|---|---|
| A baseline_scratch_256 | 旧 scratch baseline |
| B resnet34 pretrained 512 random crop | 预训练 Encoder + 更大 patch |
| C 512 multiscale | 在 B 上加入六尺度训练 |
| D 512 class-aware | 在 C 上加入 0.5 概率类别感知裁块 |
| E best + multi-scale test | 只对最佳 checkpoint 做最终 TTA |

先看 B 能否显著超过 0.3457，再逐项加入 C/D/E。若未到 0.45，应比较每类 IoU、
混淆矩阵、训练/验证曲线和 Rural/Urban 固定预测图，判断是 road/building 小目标、
barren 混淆、欠拟合还是域差异。不得换验证子集、改变 ignore 规则或改变 mIoU 定义。

当前代码只通过短时结构、损失、标签和单步显存测试；没有执行 15000 iterations，
因此没有新的正式 full-Val mIoU，也不声称达到 0.45。

## 本地网页演示

web_demo/ 提供上传单张遥感影像并查看彩色分割图、叠加图和类别占比的本地页面。网页复用 utils/inference.py 的滑窗 logits 融合，不改变模型或训练流程。

当前训练或验证结束后使用 GPU 启动：

    source ~/venvs/unet3/bin/activate
    pip install -r requirements.txt
    bash scripts/run_web_demo.sh runs/resnet34_pretrained_512_randomcrop/best_model.pth cuda:0

然后在 Windows 浏览器访问 http://127.0.0.1:7860。脚本默认使用 CPU；完整说明见 web_demo/README.md。
