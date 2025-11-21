cmake_minimum_required(VERSION 3.18)
project(TensorRTInfer LANGUAGES CXX CUDA)

# 设置 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Jetson 上必须使用旧 ABI（与系统 libtorch / libnvinfer 一致）
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# 查找 CUDA
find_package(CUDA REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72)  # Orin: SM 7.2 (Jetson AGX Orin), or 87 for Xavier
endif()

# TensorRT 路径（Jetson 默认安装位置）
set(TRT_ROOT /usr/lib/aarch64-linux-gnu)

# 包含目录
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TRT_ROOT}/include
)

# 链接库目录
link_directories(
    ${TRT_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 源文件
set(SOURCES infer.cpp)

# 可执行文件
add_executable(infer ${SOURCES})

# 链接库
target_link_libraries(infer
    nvinfer
    nvinfer_plugin
    cudart
    cublas
    cudnn
)

# 设置运行时库路径（可选）
set(CMAKE_INSTALL_RPATH "$ORIGIN")