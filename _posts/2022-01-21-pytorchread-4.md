---
layout: post
title: PyTorch源码阅读笔记（4）：自动微分张量库 —— 概览
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---
自动微分张量库是 PyTorch 的重要功能，本篇从整体流程的角度看一下相关实现。  
——————  

## API
自动微分 C++ api 示例如下:
```cpp
#include <torch/torch.h>
#include <iostream>

int main() {
  auto x = torch::ones({2, 2}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(true));
  auto y = x * 2;
  auto out = y.sum();
  out.backward();
  std::cout << x.grad() << std::endl;
}
```

## 张量存储
张量接口定义可以在 aten/src/ATen/core/TensorBody.h 看到，Tensor 类含有大量自动生成的代码，可以进行算子调用。  
Tensor 类继承自 TensorBase 类，张量相关的大量函数调用自父类 TensorBase ，TensorBase 类有一个关键的成员变量：
```cpp
protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
```
TensorImpl 类为张量的底层表示，包含了实际的数据指针和用以描述张量的元数据，它继承自 c10::intrusive_ptr_target，intrusive_ptr_target 是 c10 模块的侵入式指针模块。  
PyTorch 实现了一个侵入式指针来替代 C++ 的 shared_ptr，shared_ptr 使用时需要创建单独的对象进行引用计数，而侵入式指针在使用的类中进行引用计数，所以侵入式指针具有更好的性能。  
使用侵入式指针的类都需要实现引用计数的函数，在这里则是都需要继承 c10::intrusive_ptr_target 类，intrusive_ptr_target 有如下两个成员变量，refcount_ 记录引用计数，weakcount_ 记录弱引用计数，弱引用计数可以处理循环引用的问题：  
```cpp
  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;
```
TensorImpl 有一个 Storage 类的成员变量，Storage 有如下成员变量：
```cpp
 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
```
StorageImpl 继承了 c10::intrusive_ptr_target, 是实质上的底层数据类，保存了原始数据指针，对于 Storage 类的设计官方备注是继承自原始的 Torch7 项目，倾向于去掉此模块的设计，但是比较麻烦没人有空做。


## Variable 与 Tensor
在较新版本的 PyTorch 中，Variable 与 Tensor 进行了合并，有如下的命名空间定义：
```cpp
using torch::autograd::Variable = at::Tensor
```
不过由于兼容性考虑，并没有完全去掉 Variable 相关的 api 。

## 自动梯度
开头的示例中，backward 函数的调用会进行反向传播：
```cpp
  void backward(const Tensor & gradient={}, c10::optional<bool> retain_graph=c10::nullopt, bool create_graph=false, c10::optional<TensorList> inputs=c10::nullopt) const {
    // NB: Adding this wrapper to _backward here because we'd like our
    // 'backwards' api to accept the 'inputs' argument optionally. Since code gen
    // currently does not support optional of TensorList our approach is to replace
    // backward in native_functions.yaml with _backward and call it here instead.
    if (inputs.has_value()) {
      TORCH_CHECK(inputs.value().size() > 0, "'inputs' argument to backward cannot be empty")
      this->_backward(inputs.value(), gradient, retain_graph, create_graph);
    } else {
      this->_backward({}, gradient, retain_graph, create_graph);
    }
  }


void Tensor::_backward(TensorList inputs,
        const c10::optional<Tensor>& gradient,
        c10::optional<bool> keep_graph,
        bool create_graph) const {
  return impl::GetVariableHooks()->_backward(*this, inputs, gradient, keep_graph, create_graph);
}


namespace at { namespace impl {
namespace {
VariableHooksInterface* hooks = nullptr;
}
void SetVariableHooks(VariableHooksInterface* h) {
  hooks = h;
}
VariableHooksInterface* GetVariableHooks() {
  TORCH_CHECK(hooks, "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return hooks;
}
}} // namespace at::impl
```
VariableHooksInterface 类定义了一些虚函数，包括：
```cpp
 virtual void _backward(const Tensor&, TensorList, const c10::optional<Tensor>&, c10::optional<bool>, bool) const = 0;
```
VariableHooks 结构体继承自 VariableHooksInterface ，有如下函数：
```cpp
void VariableHooks::_backward(
    const Tensor& self,
    at::TensorList inputs,
    const c10::optional<Tensor>& gradient,
    c10::optional<bool> keep_graph,
    bool create_graph) const {
  // TODO torch::autograd::backward should take the c10::optional<Tensor> gradient directly
  // instead of us having to unwrap it to Tensor _gradient here.
  Tensor _gradient = gradient.has_value() ? *gradient : Tensor();
  std::vector<torch::autograd::Variable> input_vars(inputs.begin(), inputs.end());
  torch::autograd::backward({self}, {_gradient}, keep_graph, create_graph, input_vars);
}
```
反向传播过程来到了命名空间 torch::autograd 下。