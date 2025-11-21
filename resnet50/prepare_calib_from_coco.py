#!/usr/bin/env python3
"""
从本地已下载的 COCO val2017 中抽取 N 张图像作为 TensorRT 校准数据集
要求目录结构：
    coco/
    └── val2017/
        ├── 000000000139.jpg
        ├── 000000000285.jpg
        └── ...
"""

import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="从本地 COCO val2017 抽取校准图像")
    parser.add_argument("--coco_root", default="./coco", help="COCO 数据集根目录 (默认: ./coco)")
    parser.add_argument("--output_dir", default="./calib_data", help="输出校准数据目录 (默认: ./calib_data)")
    parser.add_argument("--num_images", type=int, default=1000, help="抽取图像数量 (默认: 1000)")
    args = parser.parse_args()

    val2017_dir = os.path.join(args.coco_root, "val2017")
    if not os.path.exists(val2017_dir):
        raise FileNotFoundError(f"未找到 val2017 目录: {val2017_dir}")

    # 获取所有 jpg 文件
    all_images = [f for f in os.listdir(val2017_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    if len(all_images) == 0:
        raise RuntimeError(f"{val2017_dir} 中没有找到 jpg 图像")

    print(f"🔍 在 {val2017_dir} 中找到 {len(all_images)} 张图像")
    
    num_to_select = min(args.num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 复制图像
    print(f"📦 正在复制 {num_to_select} 张图像到 {args.output_dir} ...")
    for img in selected_images:
        src = os.path.join(val2017_dir, img)
        dst = os.path.join(args.output_dir, img)
        shutil.copy(src, dst)
    
    print(f"✅ 完成！校准数据集已保存至: {os.path.abspath(args.output_dir)}")
    print(f"   共 {len(selected_images)} 张图像，可用于 TensorRT INT8 校准")

if __name__ == "__main__":
    main()