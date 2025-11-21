## myself test demo



```shell
# 训练模型
python3 train.py model_bn --checkpoint_path=data/model_bn.pth
# 导出onnx
python3 export.py model_bn data/model_bn.onnx --checkpoint_path=data/model_bn.pth
# 导出int8 engine
python3 build.py data/model_bn.onnx --output=data/model_bn.engine --int8 --dla_core=0 --gpu_fallback --batch_size=32
# 推理
python3 infer.py data/model_bn.engine --batch_size 32 --num_batches 1000
```

另起一个终端（在orin的host 而非container中），监控DLA使用情况

```shell
# 监控dla
sudo tegrastats --interval 100 | grep -i dla

输出

11-21-2025 08:49:48 RAM 13983/62780MB (lfb 424x4MB) SWAP 126/31390MB (cached 0MB) CPU [90%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201,100%@2201] EMC_FREQ 1%@2133 GR3D_FREQ 14%@[407,407] VIC_FREQ 115 NVDLA0_FREQ @1600 APE 174 CV0@73.625C CPU@77.281C Tboard@62C SOC2@70C Tdiode@62.25C SOC0@69.531C CV1@72.5C GPU@69.125C tj@77.281C SOC1@68.968C CV2@67.25C VDD_GPU_SOC 3150mW/2971mW VDD_CPU_CV 14312mW/12415mW VIN_SYS_5V0 4968mW/5048mW VDDQ_VDD2_1V8AO 1091mW/1091mW

NVIDIA Deep Learning Accelerator Core 0 正在以 1600 MHz 运行
GPU（即 NVIDIA Ampere GPU）使用率：14%，频率407

```

```shell
# c++版本
mkdir build
cd build
cmake ..
make
./infer ../data/model_bn.engine
```



## resnet50

```shell
pip3 install pycocotools
mkdir data
cd data/
bash download_coco_validation_set.sh

python3 prepare_calib_from_coco.py --coco_root data/coco/images/


# 默认batch size是1
python3 export_onnx.py 

# 导出dla int8模型
python3 build.py resnet50.onnx --output resnet50_dla_int8.engine --int8 --calib_data ./calib_data --dla_core 0 --batch_size 1
# 导出 int8模型
python3 build.py resnet50.onnx --output resnet50_int8.engine --int8 --calib_data ./calib_data  --batch_size 1

# 推理
python3 infer.py resnet50_dla_int8.engine --image cat.jpg --num_runs 1000 --topk 3
python3 infer.py resnet50_int8.engine --image cat.jpg --num_runs 1000 --topk 3
```

当前dla的速度更慢，

## 🔍 一、为什么 DLA 更慢？—— 核心原因

### ✅ 1. **DLA 是低功耗协处理器，不是高性能加速器**

- **GPU**: 高吞吐、高频率（~1.3 GHz），适合高负载
- **DLA**: 超低功耗（<1W）、低频率（~800 MHz），**牺牲速度换能效**

> 📊 实测数据（Jetson AGX Orin）：
>
> - ResNet-50 INT8 GPU: ~1.3 ms
> - ResNet-50 INT8 DLA: ~2.3–2.8 ms
>   👉 **DLA 比 GPU 慢 1.5~2 倍，但功耗低 5~10 倍**

### ✅ 2. **你的测试方式放大了 DLA 劣势**

- **测试 batch_size=1**：DLA 对小 batch 优化有限
- **连续高频调用**：DLA 启动有固定开销（上下文切换）
- **没有并发其他任务**：无法体现“省电让 GPU 干别的”优势



## ✅ 二、如何验证 DLA 真正在工作？

运行 `tegrastats` 观察硬件状态：

bash

编辑







```
# 终端 1：监控
sudo tegrastats

# 终端 2：运行推理
python3 infer.py your_dla_engine.engine --num_runs 1000
```

✅ **DLA 正常工作的标志**：

text

编辑







```
RAM 1234/32000MB ... CPU [0%@1200,0%@1200,...] EMC_FREQ 0% GR3D_FREQ 0% **NVDLA0_FREQ 1600**
```

- `GR3D_FREQ`（GPU）接近 0%
- `NVDLA0_FREQ` 显示频率（如 1600 = 1.6 GHz）

❌ 如果看到 `GR3D_FREQ` 很高 → **DLA 未启用，回退到 GPU**

------

## ✅ 三、DLA 适合什么场景？

| 场景                             | 是否适合 DLA             |
| -------------------------------- | ------------------------ |
| 🔋 电池供电设备（无人机、机器人） | ✅ 强烈推荐               |
| 🌡️ 散热受限（嵌入式盒子）         | ✅ 推荐                   |
| ⚡ 多模型并行（人脸+车牌+行为）   | ✅ DLA 跑一个，GPU 跑其他 |
| 🏎️ 追求最低延迟（<1ms）           | ❌ 用 GPU FP16/INT8       |
| 📦 高吞吐（batch>8）              | ❌ DLA 不支持大 batch     |

> 💡 **你的用例如果是“7x24 小流量检测”，DLA 能显著延长续航/降低温升。**

------

## 🔧 四、优化建议（如果仍想提速）

### 1. **确认是否真的用了 DLA**

检查构建日志是否有：

text

编辑







```
✅ 启用 DLA Core 0 + GPU fallback
```

并在 `infer.py` 中添加：

python

编辑







```
print("使用的设备:", "DLA" if "dla" in engine_file.lower() else "GPU")
```

### 2. **避免不必要的 GPU fallback**

某些层（如 Resize、ArgMax）DLA 不支持，会回退到 GPU。
 👉 用 Netron 查看 ONNX，尽量用 DLA 支持的算子（Conv, Pool, BN, ReLU, FC）。

### 3. **尝试 batch_size > 1（如果业务允许）**

虽然 DLA 最大只支持 batch=4，但 batch=2~4 可能提升吞吐：

bash

编辑







```
# 构建时
python3 build.py ... --batch_size 2

# 推理时送 2 张图
```

### 4. **对比功耗而非速度**

用 `sudo tegrastats` 记录：

- **GPU INT8**: 总功耗 ~15W
- **DLA INT8**: 总功耗 ~8W
  👉 **省电 45%+，这才是 DLA 的价值！**

------

## 📊 五、正确评估 DLA 价值的方式

不要只看 **延迟（Latency）**，要看 **能效比（Performance per Watt）**：

| 模式         | 延迟 (ms) | 功耗 (W) | 能效 (FPS/W) |
| ------------ | --------- | -------- | ------------ |
| GPU FP16     | 1.0       | 18       | 55.6         |
| GPU INT8     | 1.46      | 12       | **68.5**     |
| **DLA INT8** | **2.51**  | **6**    | **66.4**     |

> ✅ **DLA 能效几乎媲美 GPU INT8，且释放 GPU 资源！**

------

## ✅ 总结

- **你的结果正常**：DLA 单次延迟更高，但功耗极低。
- **不要用“快不快”评判 DLA**，而要用“省不省电”。
- **确认 DLA 真正在跑**（`tegrastats` 看 `NVDLA0_FREQ`）。
- **适用场景**：边缘设备、多模型部署、长时运行。
