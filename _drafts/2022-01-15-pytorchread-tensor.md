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

## 张量API
张量接口定义可以在 aten/src/ATen/core/TensorBody.h 看到，Tensor 类含有大量自动生成的代码，可以进行算子调用。  
Tensor 类继承自 TensorBase 类，张量相关的大量函数调用自父类 TensorBase ，TensorBase 类有一个关键的成员变量：
```cpp
protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
```
TensorImpl 类为张量的底层表示，包含了实际的数据指针和用以描述张量的元数据，它继承自 c10::intrusive_ptr_target，这里涉及到了 c10 模块的一个侵入式指针实现。

## 侵入式指针
张量库为了保证性能会尽量避免底层数据的复制，
