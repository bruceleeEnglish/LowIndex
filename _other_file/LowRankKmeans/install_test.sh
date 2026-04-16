#!/bin/bash

# 删除旧的构建目录
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# 创建新的构建目录
echo "Creating new build directory..."
mkdir -p build

# 进入构建目录
echo "Entering build directory..."
cd build

# 运行 CMake 配置
echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 编译项目
echo "Building project..."
make -j$(nproc)

# 检查编译结果
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
else
    echo "Build failed!"
    exit 1
fi