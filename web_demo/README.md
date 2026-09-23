# LoveDA 四模型横向对比 Web Demo

这个页面用于大学课程设计答辩中的定性展示。用户一次上传一张遥感影像，系统使用四个已经完成训练并经过官方提交的模型依次推理，再用统一颜色表展示四组 Mask 和 Overlay。

## 固定比较模型

1. ResNet34 Baseline
   - checkpoint：runs/resnet34_pretrained_512_randomcrop/best_model.pth
   - Hidden Test mIoU：0.457311
2. ResNet50 Cat48
   - checkpoint：runs/resnet50_pretrained_512_cat48_fast/best_model.pth
   - Hidden Test mIoU：0.446118
3. ResNet34 Diff-LR
   - checkpoint：runs/resnet34_512_diff_lr/best_model.pth
   - Hidden Test mIoU：0.454974
4. ResNet50 Cloud Final
   - checkpoint：/mnt/d/loveda-unet3plus-resnet50-cloud-final/loveda-unet3plus-resnet50-cloud_full_except_data_submission/loveda-unet3plus-resnet50-cloud/runs/cloud_resnet50_24gb_eff4_40ep/best_model.pth
   - Hidden Test mIoU：0.469813

程序启动前会逐个检查 checkpoint 是否存在，并核对 config.model 中的 backbone、cat_channels、run.name、Differential LR 标记及输出头 shape。任何模型缺失或身份不匹配时，终端会显示 [MISSING] 或 [INVALID]，服务不会使用其他权重替代。

## 公平推理协议

四个模型使用完全相同的：

- RGB 转换
- ImageNet normalization
- 单尺度 scale=1.0
- 512 × 512 sliding window
- stride=512
- horizontal flip TTA 关闭
- float32 logits 融合后 argmax
- LoveDA 七类颜色表
- Overlay alpha=0.48

checkpoint 已包含完整权重，构建模型时强制 pretrained=False、weights=None，不会重复下载 ImageNet 权重。

## 8GB GPU 顺序推理

RTX 4060 Laptop 只有 8GB 显存，程序不会让四个模型同时常驻 GPU。每个请求按固定顺序执行：

1. 加载一个 checkpoint 到 CPU；
2. 构建对应的 ResNet34/50 与 U-Net 3+；
3. 仅将当前模型移入 GPU；
4. 完成预测并保存 CPU 结果；
5. 将模型移回 CPU并删除；
6. 执行垃圾回收和 torch.cuda.empty_cache()；
7. 再加载下一个模型。

某个模型失败时，该卡片显示错误信息，其余模型继续运行。

## 启动方式

GPU 推荐命令：

    cd /home/kxcc/unet3
    source ~/venvs/unet3/bin/activate
    python -m web_demo.app --device cuda:0

浏览器访问：

    http://127.0.0.1:7860

CPU 调试：

    python -m web_demo.app --device cpu

8GB GPU 默认 patch batch 为 1。命令行也可显式指定：

    python -m web_demo.app --device cuda:0 --patch-batch-size 1

## 页面输出

- 单独展示原始 RGB 影像
- 2 × 2 四模型预测卡片
- 每张卡片可切换 Mask 与 Overlay
- 每个结果可以下载彩色 Mask、Overlay 和 LoveDA 1～7 标签图
- 统一 LoveDA 类别图例
- 模型结构、实验特征和 Hidden Test mIoU 对比表
- 每个模型的推理耗时与四模型总耗时

页面中的 Hidden Test mIoU 是各模型在 LoveDA 官方隐藏 Test 集上的总体指标，不代表当前上传图片的 IoU。上传图片没有人工 mask 时，只能定性观察预测差异，不能计算真实 IoU。

四个模型的 decoder 均为 U-Net 3+ Full-scale Skip Connection，不是普通 U-Net 或 U-Net++。
