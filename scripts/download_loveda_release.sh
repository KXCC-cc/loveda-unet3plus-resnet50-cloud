#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="KXCC-cc/loveda-unet3plus-resnet50-cloud"
TAG="dataset-v1"
ASSET_DIR="dataset_release"
mkdir -p "$ASSET_DIR"

while IFS= read -r name; do
  url="https://github.com/$REPO/releases/download/$TAG/$name"
  echo "下载 $name"
  curl --fail --location --retry 5 --continue-at - --output "$ASSET_DIR/$name" "$url"
done <<'ASSETS'
loveda_dataset.tar.part-aa
loveda_dataset.tar.part-ab-00
loveda_dataset.tar.part-ab-01
loveda_dataset.tar.part-ab-02
loveda_dataset.tar.part-ab-03
loveda_dataset.tar.part-ac
loveda_dataset.tar.part-ad
loveda_dataset.tar.part-ae
ASSETS

sha256sum --check dataset_manifest.sha256

echo "解包 LoveDA 数据集到 ./dataset"
cat "$ASSET_DIR"/loveda_dataset.tar.part-* | tar -xf -
echo "数据集准备完成。"
