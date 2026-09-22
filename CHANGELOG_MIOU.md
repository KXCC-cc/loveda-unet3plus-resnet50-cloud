# 面向 mIoU 的受控优化记录

本轮只增加可选训练策略、测试和云端保护。模型仍是 ImageNet ResNet50 Encoder、
U-Net 3+ Full-scale Skip Connections、Deep Supervision、Frozen Encoder BN 和
Decoder GroupNorm。`configs/cloud/resnet50_24gb.yaml` 保持原样，历史实验与权重未修改。

## 新增消融配置

| 配置 | 改动 | 可能受益类别 | 显存 | 时间 | baseline 公平性 |
|---|---|---|---|---|---|
| `resnet50_24gb_diff_lr.yaml` | encoder 0.001、decoder 0.01 | 全类别；降低预训练特征被早期大步更新破坏的风险 | 基本不变 | 基本不变 | 独立消融，不改 baseline |
| `resnet50_24gb_classaware.yaml` | 50% 类别感知裁块，有限 8 次后回退随机裁块 | building、road、water、barren，重点观察 barren | GPU 基本不变 | DataLoader 略增 | 改变采样分布，单独报告 |
| `resnet50_24gb_multiscale.yaml` | 0.5～1.75 六尺度缩放后裁 512 | road/building 小目标及 agricultural/forest 尺度变化 | GPU 基本不变 | CPU resize 增加 | 独立消融，不改验证集 |
| `resnet50_24gb_focal.yaml` | 0.5 CE + 1.0 Dice + 0.5 Focal，gamma=2 | 难分类像素及 barren/road 等弱类 | 小幅增加 | 小幅增加 | 独立 Loss 消融 |
| `resnet50_24gb_warmup.yaml` | 前 500 个成功 optimizer update 从 0.01 倍 LR 线性升温 | 主要改善早期稳定性，类别收益不作保证 | 不变 | 不变 | 独立调度消融 |
| `resnet50_24gb_multiscale_classaware.yaml` | 多尺度 + 类别感知 | 仅在两个单项实验有效后验证互补性 | GPU 基本不变 | DataLoader 增加 | 组合实验，不替代 baseline |

Focal 没有使用 `CE=1 + Dice=1 + Focal=1`，避免把与 CE 高度相关的监督项同时
满权重叠加。当前 0.5/1.0/0.5 只是合理起点，不能预先声称优于 CE+Dice。

## 工程改进

- poly/cosine/constant 调度器支持默认关闭的 optimizer-update 级 linear warmup。
- AMP overflow 跳过 optimizer update 时，原训练循环仍不会推进 scheduler/warmup。
- 差分学习率参数组带有 `encoder`/`decoder` 名称；真实基础 LR 写入 `config.json`。
- `summary.json` 新增 ResNet34 reference 和 `delta_vs_resnet34`。
- 显存工具支持 `--cat-channels` 和 `--loss-profile`，并输出 GPU、总显存、峰值、
  batch、crop、cat channels 与状态；OOM 不会自动缩小模型。
- 所有云训练脚本先检查 CUDA、GPU 显存、2522 张 Train 和 1669 张 Val，并核对
  image/mask 文件名后才启动训练。
- 新增 ResNet50 五头输出、Focal、warmup、差分 LR 与配置继承测试。

这些工程改进不修改标签、ignore 规则、mIoU 算法、验证集或网络宽度。

## 推荐实验顺序

1. A：`resnet50_24gb.yaml`，干净的完整 ResNet50 baseline。
2. B：`resnet50_24gb_diff_lr.yaml`。
3. C：`resnet50_24gb_classaware.yaml`。
4. D：`resnet50_24gb_multiscale.yaml`。
5. E：`resnet50_24gb_focal.yaml`。
6. Warmup 单项：`resnet50_24gb_warmup.yaml`。
7. 只组合已经由单项实验确认有效的改动；现成的首个组合配置是
   `resnet50_24gb_multiscale_classaware.yaml`。
8. 最佳 checkpoint 先做单尺度 Full Validation，再做多尺度 TTA，最后生成 Test 提交。

观察点继续使用完整 Val 的 epoch 间隔。24GB 配置 effective batch=16、每 8 个数据
轮次验证一次，约每 1260 个 optimizer updates 得到一个观察点，能够覆盖约
2500/5000/7500/10000/12500/15000 附近且不会自动早停。

## 第一轮正式云训练

第一轮仍运行未加入任何新策略的干净 baseline：

```bash
bash scripts/cloud_train_resnet50_24gb.sh
```

本轮代码修改没有执行 15000 updates 正式训练，所有收益必须由后续真实实验确认。
