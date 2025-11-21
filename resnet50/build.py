#!/usr/bin/env python3
"""TensorRT Engine Builder for Jetson Orin (with DLA + INT8 support)

Usage:
    # FP16 GPU
    python3 build.py model.onnx --output model_fp16.engine --fp16

    # INT8 DLA (推荐)
    python3 build.py model.onnx --output model_dla_int8.engine --int8 --dla_core 0 --calib_data ./calib_data --batch_size 1
"""

import argparse
import os
import numpy as np
from PIL import Image
import tensorrt as trt
import random
import pycuda.driver as cuda
import pycuda.autoinit
# 日志器：打印 TRT 信息
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ==============================
# INT8 校准器实现
# ==============================
class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_images_dir, batch_size=1, input_shape=(3, 224, 224), cache_file="calibration.cache"):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.cache_file = cache_file
        self.channel, self.height, self.width = input_shape
        
        # 获取图像路径
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [
            os.path.join(calibration_images_dir, f)
            for f in os.listdir(calibration_images_dir)
            if f.lower().endswith(supported_ext)
        ]
        if len(self.image_paths) == 0:
            raise ValueError(f"校准目录 {calibration_images_dir} 中未找到图像")
        if len(self.image_paths) < batch_size:
            raise ValueError(f"图像数量不足: {len(self.image_paths)} < {batch_size}")
        
        random.shuffle(self.image_paths)
        self.current_index = 0
        
        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # 分配 GPU 内存（固定大小）
        self.device_input = cuda.mem_alloc(trt.volume(input_shape) * batch_size * np.dtype(np.float32).itemsize)
        self.batch_data = None  # 用于暂存 CPU 数据

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.width, self.height), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        return img_array

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_paths):
            return None  # 校准结束

        batch_images = []
        for i in range(self.batch_size):
            img_path = self.image_paths[self.current_index + i]
            try:
                img = self.preprocess_image(img_path)
                batch_images.append(img)
            except Exception as e:
                print(f"⚠️ 跳过损坏图像: {img_path}")
                continue

        if len(batch_images) == 0:
            return None

        # 补齐 batch（确保数量一致）
        while len(batch_images) < self.batch_size:
            batch_images.append(batch_images[-1])

        # 合并为 batch
        self.batch_data = np.ascontiguousarray(np.stack(batch_images), dtype=np.float32)

        # 拷贝到 GPU
        cuda.memcpy_htod(self.device_input, self.batch_data)

        self.current_index += self.batch_size
        return [int(self.device_input)]  # 返回 GPU 指针！

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
# ==============================
# 构建引擎函数
# ==============================
def build_engine(onnx_file_path, engine_file_path, fp16=False, int8=False, dla_core=-1, calib_data=None, batch_size=1):
    """
    从 ONNX 构建 TensorRT 引擎
    :param onnx_file_path: 输入 ONNX 路径
    :param engine_file_path: 输出引擎路径
    :param fp16: 是否启用 FP16
    :param int8: 是否启用 INT8（需 calib_data）
    :param dla_core: DLA 核心编号（0 或 1），-1 表示不用 DLA
    :param calib_data: INT8 校准数据集路径
    :param batch_size: 静态 batch size（DLA 必须固定）
    """
    if int8 and not calib_data:
        raise ValueError("启用 --int8 必须指定 --calib_data")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = 2 * (1 << 30)  # 2 GB

        # 启用 FP16
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ 启用 FP16 精度")

        # 启用 INT8 + 校准
        if int8:
            if not builder.platform_has_fast_int8:
                raise RuntimeError("当前平台不支持 INT8")
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = ImageNetCalibrator(
                calibration_images_dir=calib_data,
                batch_size=batch_size,
                input_shape=(3, 224, 224),
                cache_file="calibration.cache"
            )
            print(f"✅ 启用 INT8 校准（数据集: {calib_data}, batch={batch_size})")

        # 启用 DLA（如果指定）
        if dla_core >= 0:
            # if builder.num_dla_cores == 0:
            #     raise RuntimeError("当前平台不支持 DLA")
            # if dla_core >= builder.num_dla_cores:
            #     raise ValueError(f"DLA 核心 {dla_core} 不存在（共 {builder.num_dla_cores} 个）")
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # 关键：不支持的层回退到 GPU
            print(f"✅ 启用 DLA Core {dla_core} + GPU fallback")

        # 解析 ONNX
        if not os.path.exists(onnx_file_path):
            raise FileNotFoundError(f"ONNX 文件不存在: {onnx_file_path}")
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ ONNX 解析失败，错误信息：")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        print(f"✅ 成功解析 ONNX，输入: {network.get_input(0).name}, 输出: {network.get_output(0).name}")

        # 构建序列化网络（推荐方式）
        print("⏳ 正在构建 TensorRT 引擎...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
            print(f"✅ 引擎已保存: {engine_file_path}")
        else:
            print("❌ 引擎构建失败！")
            return None


# ==============================
# 主函数
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX (with DLA + INT8 support)")
    parser.add_argument("onnx", help="Input ONNX file path")
    parser.add_argument("--output", "-o", required=True, help="Output TensorRT engine file path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision (requires --calib_data)")
    parser.add_argument("--calib_data", type=str, help="Path to calibration dataset directory (for --int8)")
    parser.add_argument("--dla_core", type=int, default=-1, choices=[-1, 0, 1], help="Use DLA core (0 or 1). -1 means GPU only.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (must match ONNX input if static)")
    args = parser.parse_args()

    # 检查互斥选项
    if args.fp16 and args.int8:
        raise ValueError("不能同时启用 --fp16 和 --int8")

    build_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        dla_core=args.dla_core,
        calib_data=args.calib_data,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()