# LoveDA 数据集快照

数据来自本机项目的LoveDA目录，并按原目录结构打包。云端解包后根目录为
dataset/，训练代码无需修改路径。

## 文件数量

- Train：2522张影像 + 2522张mask
- Validation：1669张影像 + 1669张mask
- Test：1796张影像
- 总文件数：10178
- 打包体积：约9.0GB
- 分卷：5个，每个不超过1.9GB

## 标签映射

LoveDA原始mask的0表示no-data，1至7表示七种地物类别。训练时统一映射为：

- 0 -> 255（ignore）
- 1至7 -> 0至6

完整性校验使用dataset_manifest.sha256。数据分卷不写入Git历史，而是上传到
dataset-v1 Release；scripts/download_loveda_release.sh负责下载、校验和解包。
