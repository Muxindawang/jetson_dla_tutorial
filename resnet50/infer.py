#!/usr/bin/env python3
"""TensorRT Inference Script for Jetson Orin (with DLA support)
Supports ResNet-18/50 etc. exported from torchvision.

Usage:
    # 推理单张图像
    python3 infer.py model.engine --image cat.jpg
    # 随机数据推理（性能测试）
    python3 infer.py model.engine --batch_size 8 --num_runs 100
"""

import argparse
import os
import time
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# =============================================
import pycuda.driver as cuda
import pycuda.autoinit  # 必须导入以初始化 CUDA 上下文
import tensorrt as trt

# ImageNet 类别标签（简化版，仅前 1000 类）
try:
    import json
    with open('imagenet_class_index.json', 'r') as f:
        IMAGENET_CLASSES = json.load(f)
except Exception as e:
    print(f"⚠️ 未找到 imagenet_class_index.json 或加载失败: {e}")
    IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"引擎文件不存在: {engine_path}")
    t0 = time.perf_counter()
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("反序列化引擎失败！")
    t1 = time.perf_counter()
    print(f"[⏱️] 引擎加载耗时: {(t1 - t0) * 1000:.2f} ms")
    return engine


def allocate_buffers(engine, batch_size=1):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    t0 = time.perf_counter()
    for binding in engine:
        shape = list(engine.get_binding_shape(binding))
        if shape[0] == -1:
            shape[0] = batch_size
        size = trt.volume(shape) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem, 'shape': tuple(shape)})
        else:
            outputs.append({'host': host_mem, 'device': device_mem, 'shape': tuple(shape)})
    t1 = time.perf_counter()
    print(f"[⏱️] 缓冲区分配耗时: {(t1 - t0) * 1000:.2f} ms")
    return inputs, outputs, bindings, stream


def preprocess_image(image_path, target_size=(224, 224)):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return img_array


def run_inference(context, bindings, inputs, outputs, stream, input_data):
    np.copyto(inputs[0]['host'], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    return outputs[0]['host'].copy()


def main():
    parser = argparse.ArgumentParser(description="TensorRT Inference with DLA support")
    parser.add_argument("engine", help="Path to TensorRT engine file (.engine)")
    parser.add_argument("--image", "-i", help="Path to input image (for single inference)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for random data test")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for performance test")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    args = parser.parse_args()

    total_start = time.perf_counter()

    # 加载引擎
    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    if not context:
        raise RuntimeError("无法创建执行上下文")

    # 获取输入 shape
    input_shape = engine.get_binding_shape(0)
    if input_shape[0] == -1:
        input_shape = (args.batch_size,) + input_shape[1:]
    _, channels, height, width = input_shape
    print(f"✅ 检测到模型输入 shape: {input_shape}")

    # 分配缓冲区
    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=args.batch_size)

    # 准备输入数据
    prep_start = time.perf_counter()
    if args.image:
        if args.batch_size != 1:
            print("⚠️ 指定了图像，强制 batch_size=1")
            args.batch_size = 1
        input_data = preprocess_image(args.image, (height, width))
        input_data = np.expand_dims(input_data, axis=0)
        print(f"🖼️ 加载图像: {args.image}")
    else:
        print(f"🎲 使用随机数据 (batch_size={args.batch_size})")
        input_shape = (args.batch_size, channels, height, width)
        input_data = np.random.randn(*input_shape).astype(np.float32)
    prep_end = time.perf_counter()
    print(f"[⏱️] 输入准备耗时: {(prep_end - prep_start) * 1000:.2f} ms")

    # 预热
    print("🔥 预热中...")
    warmup_start = time.perf_counter()
    for _ in range(50):
        run_inference(context, bindings, inputs, outputs, stream, input_data)
    warmup_end = time.perf_counter()
    print(f"[⏱️] 预热耗时 (5 次): {(warmup_end - warmup_start) * 1000:.2f} ms")

    # 正式推理
    print(f"🚀 开始正式推理 ({args.num_runs} 次)...")
    infer_start = time.perf_counter()
    for _ in range(args.num_runs):
        output = run_inference(context, bindings, inputs, outputs, stream, input_data)
    infer_end = time.perf_counter()

    total_end = time.perf_counter()

    # 计算性能指标
    total_time_ms = (total_end - total_start) * 1000
    infer_time_ms = (infer_end - infer_start) * 1000
    avg_latency_ms = infer_time_ms / args.num_runs
    throughput = (args.batch_size * args.num_runs) / (infer_time_ms / 1000)  # imgs/sec

    # 输出结果
    if args.image:
        probs = output.reshape(-1)
        top_indices = np.argsort(probs)[-args.topk:][::-1]
        print(f"\n🎯 Top-{args.topk} 预测结果:")
        for i, idx in enumerate(top_indices):
            class_name = IMAGENET_CLASSES.get(str(idx), IMAGENET_CLASSES[idx]) if isinstance(IMAGENET_CLASSES, dict) else IMAGENET_CLASSES[idx]
            print(f" {i+1}. {class_name} (prob={probs[idx]:.4f})")
    else:
        print(f"\n📊 性能统计 (batch_size={args.batch_size}, runs={args.num_runs}):")
        print(f"   平均延迟: {avg_latency_ms:.2f} ms")
        print(f"   吞吐量:   {throughput:.1f} images/sec")

    # 打印完整耗时摘要
    print("\n" + "="*50)
    print("[⏱️] 耗时汇总:")
    print(f"   引擎加载:      已在 load_engine 中打印")
    print(f"   缓冲区分配:    已在 allocate_buffers 中打印")
    print(f"   输入准备:      {(prep_end - prep_start) * 1000:.2f} ms")
    print(f"   预热 (5 次):   {(warmup_end - warmup_start) * 1000:.2f} ms")
    print(f"   正式推理 ({args.num_runs} 次): {infer_time_ms:.2f} ms")
    print(f"   —— 平均每次:   {avg_latency_ms:.2f} ms")
    print(f"   总耗时 (端到端): {total_time_ms:.2f} ms")
    print("="*50)

    print("\n✅ 推理完成！请配合 `sudo tegrastats` 确认 DLA 是否工作。")


if __name__ == "__main__":
    main()