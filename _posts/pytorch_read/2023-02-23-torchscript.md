---
title: "PyTorch 源码阅读笔记（5）：TorchScript"
categories: [PyTorch 源码阅读笔记]
description: TorchScript
keywords: 
- PyTorch 源码阅读
- TorchScript
tags: [PyTorch]
date: 2023-02-23
draft: false
---

## TorchScript 的使用
python api:
```py
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h
    
scripted_module = torch.jit.script(MyCell().eval())
```
C++ api:
```cpp
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "ok\n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}
```
## 关于 TorchScript
细节参考：[TorchScript](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/csrc/jit/OVERVIEW.md)  
TorchScript 实现的功能，从使用角度看是用 Python 编写模型，然后在 C++ 内运行，大致有如下步骤：  
1. 解析 Python 代码为抽象语法树
2. 转化语法树为模型中间表示 IR
3. 根据 IR 生成模型
4. 执行模型（根据运行时信息优化模型-JIT）  

从流程上看，PyTorch 在 C++ 端（LibTorch）实现了一个编译器，编译运行了一个 Python 的子集语言，即为 TorchScript：  
1 ~ 3为编译器的前端（语法分析、类型检查、中间代码生成），4为编译器后端（代码优化、执行代码生成与优化）  
## more
1. 使用角度看，TorchScript 适用于生产部署 PyTorch 模型，不过实际工作中没有直接使用过，一般训练完成以后会选择导出 onnx，openvino等格式（导出过程其实使用了相关模块），单独部署为推理服务
2. 原理涉及较多编译原理相关内容，学习后再补充