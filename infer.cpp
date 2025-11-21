#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

// TensorRT 核心头文件
#include "NvInfer.h"
#include "NvInferPlugin.h"

// CUDA 运行时 API（用于显存管理、数据拷贝等）
#include "cuda_runtime_api.h"

// 宏定义：检查 CUDA 调用是否出错，若出错则终止程序
#define CHECK(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

using namespace nvinfer1;

/**
 * 自定义 Logger 类
 * 用于接收 TensorRT 内部的日志信息（INFO/WARNING/ERROR）
 */
// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;


/**
 * 创建 TensorRT Runtime 对象，并初始化插件（如 GroupNorm、Swish 等）
 */
// Helper: read engine file
std::unique_ptr<IRuntime> createInferRuntime() {
    // 初始化所有官方插件（即使当前模型没用，也建议调用）
    initLibNvInferPlugins(&gLogger, "");
    // 创建 Runtime，用于反序列化引擎
    return std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
}

std::vector<char> loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Cannot open engine file: " << enginePath << std::endl;
        exit(1);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取全部内容到内存
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();
    return engineData;
}

/**
 * 生成随机浮点数输入数据（仅用于演示）
 * 实际部署时应替换为真实图像预处理后的数据
 * @param data 输出缓冲区指针
 * @param size 元素个数（非字节数）
 */
// Generate random input data (for demo)
void generateRandomInput(float* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // CIFAR-10 归一化后范围约 [-2, 2]
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_file>" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    std::cout << "Loading engine: " << enginePath << std::endl;

    // Load engine
    // === 第一步：加载并反序列化 TensorRT 引擎 ===
    auto engineData = loadEngine(enginePath);
    auto runtime = createInferRuntime();
    std::unique_ptr<ICudaEngine> engine(
        runtime->deserializeCudaEngine(engineData.data(), engineData.size())
    );
    if (!engine) {
        std::cerr << "Failed to deserialize engine!" << std::endl;
        return -1;
    }

    // 创建执行上下文（用于实际推理）
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        return -1;
    }

    // Get input/output binding info
    // === 第二步：分配 GPU 输入/输出缓冲区 ===
    int numBindings = engine->getNbBindings(); // 通常为 2（1输入 + 1输出）
    std::vector<void*> buffers(numBindings);   // 存放设备指针
    std::vector<size_t> bufferSizes(numBindings); // 存放每个 buffer 的字节数

    // 遍历所有绑定（bindings）
    for (int i = 0; i < numBindings; ++i) {
        // 获取该 binding 的维度和数据类型
        auto dims = engine->getBindingDimensions(i);
        auto type = engine->getBindingDataType(i); // 通常是 DataType::kFLOAT

        // 计算总元素数（volume）
        size_t vol = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            vol *= dims.d[j];
        }

        // 假设是 float32，计算字节数
        size_t size = vol * sizeof(float);
        bufferSizes[i] = size;

        // 在 GPU 上分配显存
        CHECK(cudaMalloc(&buffers[i], size));

        // 打印绑定信息（调试用）
        if (engine->bindingIsInput(i)) {
            std::cout << "输入绑定 " << i << ": ";
        } else {
            std::cout << "输出绑定 " << i << ": ";
        }
        std::cout << "shape=(";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j == dims.nbDims - 1 ? "" : ", ");
        }
        std::cout << "), 大小=" << size << " 字节" << std::endl;
    }

    // === 第三步：准备主机（CPU）端输入/输出数据 ===
    std::vector<float> inputHost(bufferSizes[0] / sizeof(float));  // 输入数据（CPU）
    std::vector<float> outputHost(bufferSizes[1] / sizeof(float)); // 输出数据（CPU）

    // 用随机数据填充输入（实际应用中应替换为真实图像）
    generateRandomInput(inputHost.data(), inputHost.size());

    // 将输入数据从 CPU 拷贝到 GPU
    CHECK(cudaMemcpy(buffers[0], inputHost.data(), bufferSizes[0], cudaMemcpyHostToDevice));

    // === 第四步：预热（warm-up）推理，避免首次延迟干扰测量 ===
    for (int i = 0; i < 10; ++i) {
        // enqueueV2 是异步推理接口，需配合 cudaStream 使用（此处用默认流 0）
        context->enqueueV2(buffers.data(), 0, nullptr);
    }
    cudaDeviceSynchronize(); // 等待 GPU 完成所有任务

    // === 第五步：正式计时推理 ===
    const int numRuns = 100000; // 推理次数
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numRuns; ++i) {
        context->enqueueV2(buffers.data(), 0, nullptr);
    }
    cudaDeviceSynchronize(); // 确保所有推理完成
    auto end = std::chrono::high_resolution_clock::now();

    // 计算总耗时（微秒）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 将输出结果从 GPU 拷贝回 CPU
    CHECK(cudaMemcpy(outputHost.data(), buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost));

    // === 第六步：输出结果与性能统计 ===
    std::cout << "\n前 10 个输出 logits:\n";
    for (int i = 0; i < std::min(10, (int)outputHost.size()); ++i) {
        std::cout << outputHost[i] << " ";
    }
    std::cout << std::endl;

    // 计算平均延迟（毫秒）和吞吐量（样本/秒）
    float avgLatencyMs = static_cast<float>(duration) / numRuns / 1000.0f;
    // 注意：这里假设每次推理处理一个 batch（如 batch=32，则 throughput = 32 / latency）
    float throughput = 1000.0f / avgLatencyMs;

    std::cout << "\n=== 性能统计 ===\n";
    std::cout << "平均延迟: " << avgLatencyMs << " ms\n";
    std::cout << "吞吐量:   " << throughput << " 样本/秒\n";

    // === 第七步：释放 GPU 显存 ===
    for (auto& buf : buffers) {
        if (buf) {
            cudaFree(buf);
        }
    }

    std::cout << "推理完成。\n";
    return 0;
}