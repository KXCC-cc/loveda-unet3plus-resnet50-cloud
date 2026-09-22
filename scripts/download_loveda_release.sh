#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="KXCC-cc/loveda-unet3plus-resnet50-cloud"
TAG="dataset-v1"
ASSET_DIR="dataset_release"
mkdir -p "$ASSET_DIR"

for suffix in aa ab ac ad ae; do
  name="loveda_dataset.tar.part-\${suffix}"
  url="https://github.com/\${REPO}/releases/download/\${TAG}/\${name}"
  echo "下载 \${name}"
  curl --fail --location --retry 5 --continue-at - --output "\${ASSET_DIR}/\${name}" "$url"
done

if [ -f dataset_manifest.sha256 ]; then
  sha256sum --check dataset_manifest.sha256
fi

echo "解包 LoveDA 数据集到 ./dataset"
cat "\${ASSET_DIR}"/loveda_dataset.tar.part-* | tar -xf -
echo "数据集准备完成。"
