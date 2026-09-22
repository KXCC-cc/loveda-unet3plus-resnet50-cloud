# 给 Antigravity 的执行提示词（旧 baseline，禁止直接用于新 ResNet 实验）

> 当前 ResNet34/50 U-Net 3+ 的运行方式以 `README.md` 和 `scripts/` 为准。

下面的提示词用于用户随后粘贴给 Antigravity；文件本身没有执行任何检查或训练。

## 环境纠正：优先于旧提示、旧安装计划和自动继续任务

只使用已有解释器 `D:\Anaconda\envs\Buildingfire_Project\python.exe`。
用户指定 Python 3.9.25；该环境中已找到 `torch-2.5.1+cu121.dist-info`，与用户提供的包列表一致。
它不是 `D:\unet3\.venv\Scripts\python.exe`。不能把项目 `.venv` 的缺包情况当作指定环境缺包。

立即停止你此前为本项目发起的 PyTorch 下载、pip dry-run、安装和后续自动安装任务（包括 torch 2.6.0+cu124 计划），仅取消你自己能明确识别的任务，不结束无关进程。
禁止下载、安装、升级、降级或卸载依赖，禁止新建/重建虚拟环境，禁止执行 `pip install -r requirements.txt`。
下面的运行授权只包括使用现有环境执行检查与训练，不包括改变环境。

---

你负责监督运行我的 PyTorch U-Net 3+ + LoveDA 七分类项目。项目根目录为 D:\unet3，Windows 环境。当前阶段不制作网页或 UI。

本请求授权你按“实际阅读 → 环境预检 → 有上限的短检查 → 正式 GPU 训练 → 展示样本推理”顺序执行。必须先取得每一步的真实结果，不能直接启动完整训练，不能凭静态检查声称运行成功。

一、先实际阅读与保护数据

1. 阅读 README.md、requirements.txt、train.py、test_model.py、predict.py、models 和 utils 中的源码，以当前文件为准。修改、日志、检查脚本和输出全部限制在 D:\unet3。
2. dataset 原始图像和标签不得修改、移动或删除。demo_assets\Rural 和 demo_assets\Urban 各有 20 张展示副本；核对 manifest.json，保留它们，不纳入训练、验证、早停、调参或类别权重统计。
3. Test 没有真实 mask。展示预测不是 GT，不得编造这些图片的 Dice/mIoU。需要真值对照时只用标注清楚的 Val 样本。

二、固定使用已有 Python 3.9.25 环境

4. 工作目录设为 D:\unet3，所有 Python 命令固定使用 D:\Anaconda\envs\Buildingfire_Project\python.exe，不能依赖终端自动激活状态、裸 python、裸 pip 或编辑器默认选中的 .venv。先用该解释器打印 sys.executable、sys.version、torch.__file__、torch.__version__、torch.version.cuda、torch.cuda.is_available() 以及 numpy/Pillow/tqdm 的实际导入版本。用户提供的包列表为 torch 2.5.1+cu121、numpy 2.0.1、Pillow 11.3.0、tqdm 4.67.3；这些是已有版本记录，不是让你重装的清单。
5. 继续只读核对 GPU 型号、显存、驱动和磁盘空间。若导入或 CUDA 检查出错，先报告完整解释器路径和原始错误，不安装其他 torch、不切换环境、不降级 NumPy、不重建 .venv。不能为了匹配旧版 PyTorch 的泛化依赖建议破坏现有环境。
6. 后续结构检查、短检、正式训练、恢复和推理都使用同一个上述绝对路径。GPU 不可用时停止并报告，不悄悄退回 CPU。可使用以下 PowerShell 命令进行第一步核对（这不是安装命令）：

```powershell
Set-Location 'D:\unet3'
& 'D:\Anaconda\envs\Buildingfire_Project\python.exe' -c "import sys, torch; print('Python:', sys.executable); print(sys.version); print('Torch path:', torch.__file__); print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

对应结构检查与训练命令必须保持同一解释器，例如：

```powershell
& 'D:\Anaconda\envs\Buildingfire_Project\python.exe' 'D:\unet3\test_model.py' --device cuda:0 --batch-size 1 --image-size 256 256
# 只有后面的真实数据短检通过后，才执行正式训练：
& 'D:\Anaconda\envs\Buildingfire_Project\python.exe' 'D:\unet3\train.py' --device cuda:0
```

三、先做短检查，不能把完整一轮当预检

7. 首先执行现有 test_model.py，显式指定 cuda:0、batch-size 1、image-size 256 256。检查 main 和四个 aux 都是 [1,7,256,256]。注意该脚本是 eval/no_grad、未启用 AMP，成功只能证明相应的结构检查，不能证明反传和训练成功。
8. 当前 train.py 没有已确认的 smoke/max-steps 参数，不得编造不存在的命令。需要时新增直观的独立短检查脚本，复用现有 Dataset、模型、损失和 checkpoint 逻辑；仅用真实 Train 数据做 2～5 次参数更新，验证标签 long/ignore=255、五输出、CE+Dice、AMP/GradScaler、梯度有限、参数实际变化及峰值显存。
9. 再用最多 2 张 Val 全图（Rural/Urban 各一张）检查滑窗覆盖、logits 融合、指标及验证损失；独立检查 checkpoint 保存、读取和继续一次参数更新，包含 optimizer、scheduler、scaler。预检产物写 runs\preflight_日期时间，不能污染正式最佳权重。短检指标只代表流程检查，不是完整验证结果。
10. 不直接用 --epochs 1 代替短检查：当前默认一轮有 10088 个训练裁块，完整 Val 有 1669 张，默认每张 49 个滑窗。遇到错误，保留堆栈、做最小必要修复并重新检查，不随意重构 Decoder、不更换 loss 或降低验证标准来掩盖问题。

四、预检通过后正式训练并保存证据

11. 预检通过后，先报告实际 GPU、短检结果、显存情况、拟用参数及准确命令，再进入正式训练。起点为：cuda:0、AMP 开启、256×256 原分辨率 crop、batch size 2、num_workers 0、每图每轮 4 次取样、50 epochs、AdamW lr=3e-4/weight_decay=1e-4、CosineAnnealingLR、CE+Dice。aux 权重为 0.5/0.25/0.125/0.0625，默认不加类别权重和 Focal。若 batch 2 不适合实际显存，在正式开始前改为 1 并记录；不要未经验证改成大尺寸。
12. 原始 Train/Val 用于正式训练和验证，展示样本保持隔离。默认整图 Val stride=128×128；可报告其计算量，但不能在同一实验中悄悄改变 stride 或改成 resize 后继续比较 best mIoU。
13. 使用新的独立实验目录保存日志和 checkpoint，保留精确命令、环境版本、配置和源码版本记录。记录每轮 train loss、val main loss、mIoU、Dice、逐类指标、学习率及耗时；如果当前只有控制台输出，先补简单 JSONL/CSV 记录，方便以后画训练曲线，保持训练算法不变。
14. 同时只启动一个正式训练进程。监督真实进程与日志，保留错误信息；遇到 OOM、非有限损失、连续 AMP 溢出或进程异常退出时停止盲目重试，说明原因并处理。长时间验证不直接认定为卡死。不要改系统休眠设置；提醒用户接通电源、避免睡眠。说明当前会话/进程的实际监控边界，不承诺关闭工具后仍会永久监控。
15. 最佳权重根据完整 Val mIoU 保存。恢复正式实验必须沿用 checkpoint 中的 epochs、batch、crop、stride、增强和 loss 等配置。不能把 epochs=1 的短检权重直接改为 epochs=50 续训，也不能忽略旧四输出模型的缺失键。正式训练从头开始；正式中断时才使用该实验自己的完整 checkpoint 恢复。

五、训练后准备展示结果，但不做界面

16. 正式训练完成后，使用该实验的 best_model.pth 为 demo_assets 的 40 张图片生成预测。按 Rural/Urban 分目录保存到 predictions\demo_实验名，保持原图尺寸，先融合/插值 logits，再 argmax；避免跨域同名文件覆盖。
17. 保存原始类别索引预测图及来源、所用 checkpoint、图像尺寸等清单，便于以后制作彩色图与叠加展示。本次不要开发页面，也不要依据这 40 张 Test 展示图选择超参数。
18. 最终汇报：真实执行了什么、是否修复代码、环境版本、训练配置、最佳 epoch/完整验证 mIoU/Dice、日志/权重/40 张预测位置，以及尚未解决的问题。只引用真实日志，明确区分未执行、失败、短检通过和正式训练完成。

请先取消此前误选 .venv 导致的下载/安装计划，再用指定的现有解释器完成只读核对，然后继续短检和训练流程，逐阶段汇报关键结果。

---

AMP 接口参考：https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html 。本项目当前复用既有环境，不需要因文档出现新版本而升级或安装。
