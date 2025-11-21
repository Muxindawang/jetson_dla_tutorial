#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# =============== 兼容 NumPy 1.20+ ===============
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
# =============================================

import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torchvision
import torchvision.transforms as transforms
import time


def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine):
    """分配缓冲区（显式 batch 模式，不乘 batch_size）"""
    inputs = []
    outputs = []
    bindings = []

    for binding in engine:
        shape = engine.get_binding_shape(binding)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        size = trt.volume(shape)  # ✅ 不再乘 batch_size
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs):
    for inp in inputs:
        cuda.memcpy_htod(inp["device"], inp["host"])
    context.execute_v2(bindings=bindings)
    for out in outputs:
        cuda.memcpy_dtoh(out["host"], out["device"])
    return [out["host"].reshape(out["shape"]) for out in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", help="Path to the TensorRT engine (.engine)")
    parser.add_argument("--dataset_path", type=str, default="./data/cifar10", help="CIFAR-10 dataset root")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (must match engine)")
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to run (default: all)")
    args = parser.parse_args()

    # 1. 加载引擎
    print(f"Loading engine: {args.engine}")
    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    # 2. 分配缓冲区
    inputs, outputs, bindings = allocate_buffers(engine)
    input_shape = inputs[0]["shape"]
    print(f"Expected input shape: {input_shape}")

    # 3. 准备 CIFAR-10 测试集（与 build.py 一致的 transform）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # 4. 推理 + 准确率统计
    total_correct = 0
    total_samples = 0
    total_time = 0.0

    num_batches = args.num_batches or len(test_loader)
    print(f"Running inference on {min(num_batches * args.batch_size, len(test_dataset))} samples...")

    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break

        # 转为 numpy 并确保形状匹配
        images_np = images.numpy().astype(np.float32)
        actual_batch = images_np.shape[0]

        # 检查是否需要填充（最后一 batch 可能不足）
        if actual_batch != input_shape[0]:
            print(f"Warning: last batch size {actual_batch} != engine batch {input_shape[0]}. Skipping.")
            continue

        # 填充到 host buffer
        np.copyto(inputs[0]["host"], images_np.ravel())

        # 推理
        start = time.perf_counter()
        output = do_inference(context, bindings, inputs, outputs)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # 计算准确率
        pred = np.argmax(output[0], axis=1)
        correct = (pred == labels.numpy()).sum()
        total_correct += correct
        total_samples += actual_batch

        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}/{num_batches}, Acc: {correct/actual_batch:.2%}")

    # 5. 输出结果
     # 防止除零错误
    if total_time <= 0:
        total_time = 1e-9  # 1 纳秒作为下限

    avg_latency_ms = (total_time / num_batches) * 1000
    throughput = total_samples / total_time
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*50)
    print(f"✅ Inference completed!")
    print(f"   Total samples: {total_samples}")
    print(f"   Accuracy:      {accuracy:.2%}")
    print(f"   Avg latency:   {avg_latency_ms:.3f} ms")
    print(f"   Throughput:    {throughput:.2f} samples/sec")
    print("="*50)

    print("\n" + "="*50)
    print(f"✅ Inference completed!")
    print(f"   Total samples: {total_samples}")
    print(f"   Accuracy:      {accuracy:.2%}")
    print(f"   Avg latency:   {avg_latency_ms:.2f} ms")
    print(f"   Throughput:    {throughput:.2f} samples/sec")
    print("="*50)


if __name__ == "__main__":
    main()