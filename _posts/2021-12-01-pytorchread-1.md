---
layout: post
title: PyTorch源码阅读笔记（1）：目录结构与libtorch编译
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---

最近断断续续看了部分PyTorch源代码，整理一下零散的笔记  
——————

## PyTorch代码目录结构
参考PyTorch官方描述，大致代码结构如下所述：
* c10 - Core library —— 核心库，包含最基本的功能，aten/src/ATen/core中的代码在逐渐往此处迁移
* aten - PyTorch C++ 张量库，不包括自动梯度支持
  * aten/src - 
    * aten/src/ATen
      * aten/src/ATen/core - 核心函数库，逐步往c10迁移中。
      * aten/src/ATen/native - 原生算子库，大部分CPU算子在此一级目录下，除了一些有特殊编译需求的算子在cpu目录下
        * aten/src/ATen/native/cpu - 某些需要类似AVX等特殊指令集编译的cpu算子实现
        * aten/src/ATen/native/cuda - CUDA 算子
        * aten/src/ATen/native/sparse - CPU 和 CUDA 的稀疏矩阵算子
        * aten/src/ATen/native/mkl，aten/src/ATen/native/mkldnn，...... - 如文件夹描述，对应的算子
* torch - 实际的PyTorch库，除了torch/csrc之外的都是Python模块
  * torch/csrc - Python 和 C++ 混编库
    * torch/csrc/jit - TorchScript JIT
      frontend
    * torch/csrc/autograd - 自动微分实现
    * torch/csrc/api - The PyTorch C++ frontend.
    * torch/csrc/distributed - 
* tools - 代码生成模块，PyTorch很多代码是在编译时自动生成的
* test - Python前端单元测试模块，C++前端的单元测试在其他文件夹
* caffe2 - Caffe2 库合并入PyTorch，具体合并了哪些官方说的太抽象，以后看到了再更新

## PyTorch C++ 模块的编译
*PyTorch 官方有单独打包的 C++ 库 libtorch，参照官方提供的libtorch库编译方式*  

首先clone仓库，由于会下载很多第三方仓库，需要较长时间，并且克隆失败的库可能会导致编译失败:
```shell
git clone -b master --recurse-submodule https://github.com/PyTorch/PyTorch.git
mkdir PyTorch-build
cd PyTorch-build
```
直接使用官方的cmake文件编译就可以，即使是单独编译 C++ 模块也需要安装Python，因为有大量代码是通过Python脚本生成，看源码之前还需要把编译完成后生成的cpp文件复制到源码目录  
我使用的cmake命令：
```
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=/data/build/PyTorch-read/PyTorch-install ../PyTorch

cmake --build . --target install -j8
```

测试一下C++库的使用:
```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor1 = torch::eye(3);
  torch::Tensor tensor2 = torch::exp(tensor1);
  torch::add(tensor1, tensor2);
  std::cout << tensor2 << std::endl;
}
```
cmake配置文件（这东西写起来真的是费劲）:
```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

cmake_policy(SET CMP0074 NEW) 
set(CMAKE_PREFIX_PATH "/data/build/PyTorch-read/PyTorch-install")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp)
target_include_directories(main PUBLIC
"${TORCH_INCLUDE_DIRS}"
)
target_link_libraries(main
"${TORCH_LIBRARIES}"
"/usr/lib/x86_64-linux-gnu/libpthread.so" 
"/usr/lib/x86_64-linux-gnu/libm.so" 
"/usr/lib/x86_64-linux-gnu/libdl.so"
"/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so"  
"/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.so" 
"/opt/intel/mkl/lib/intel64/libmkl_core.so"
)
set_property(TARGET main PROPERTY CXX_STANDARD 14)
```
可能有一些库的依赖问题，安装对应的库即可。