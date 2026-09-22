# LoveDA 本地网页演示

这个界面复用项目现有的模型构建、ImageNet 归一化和滑窗推理代码。上传的图片只在当前进程内读取，不会发送到外部服务器。

## 为什么默认使用 CPU

web_demo.app 默认使用 CPU，避免在训练或完整验证仍占用 GPU 时争抢显存和算力。此时可以编写、查看代码，但不建议启动识别服务。

当前训练或验证结束后，推荐使用 GPU：

    cd /home/kxcc/unet3
    source ~/venvs/unet3/bin/activate
    pip install -r requirements.txt
    python -m web_demo.app --checkpoint runs/resnet34_pretrained_512_randomcrop/best_model.pth --device cuda:0

浏览器访问：http://127.0.0.1:7860

如果必须与 GPU 任务同时运行，可使用 CPU：

    python -m web_demo.app --checkpoint runs/resnet34_pretrained_512_randomcrop/best_model.pth --device cpu

CPU 滑窗推理会明显慢于 GPU，但不会抢占 CUDA 显存。

## 页面输出

- 原始 RGB 图片
- 固定 LoveDA 颜色表的预测图
- 原图与预测叠加图
- 七类像素占比
- 彩色 PNG 下载
- LoveDA 标签编号 1～7 的单通道 PNG 下载

推理始终先在完整 logits 上进行滑窗融合，最后才执行 argmax，不会对离散类别编号使用双线性插值。
