---
layout: post
title: PyTorch源码阅读笔记（4）：张量库
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---
PyTorch 的张量库实现有一篇核心作者之一 Edward Z Yang 的介绍：[PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)。  
截至今天有一些接口变化，本篇结合代码学习下相关实现。  
——————  

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
StorageImpl 继承了 c10::intrusive_ptr_target, 是实质上的底层数据类，保存了原始数据指针，对于 Storage 类的设计官方备注是继承自原始的 Torch7 项目，倾向于去掉此模块的设计，但是比较麻烦没人有空做（过于真实的理由）。