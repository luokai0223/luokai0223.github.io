---
layout: post
title: PyTorch源码阅读笔记（2）：operators算子——native算子调用过程
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---

最开始想要看看PyTorch源码是从自定义算子开始，所以先看看PyTorch的native算子（翻译似乎是原生算子？因为最终的具体实现都是和平台相关的，比如CUDA代码）是如何被调用的  
——————  

## 算子基本信息定义
按照官方描述，所有的原生算子（函数）都定义在aten/src/ATen/native/native_functions.yaml文件里面，以一个add算子为例：  
![算子示例-add](/assets/images/native-functions-add.png)  
配置文件中定义了算子的输入输出、支持平台等信息，具体内容官方有详细描述。   
## 算子调用过程
算子在配置文件中被定义后，在编译过程中被代码生成脚本处读取，生成多处C++代码，同样以add.out算子为例，当代码中使用此算子时，首先调用aten/src/ATen/ops/add.h 中定义的 add_out 内联函数：
![add_out-内联定义](/assets/images/add-h.png)  
函数内继续调用 aten/src/ATen/ops/add_ops.h 以及 aten/src/ATen/Operators_*.cpp 中定义的 add_out 结构体的成员函数：  
![add_ops-结构体](/assets/images/add-ops-h.png)  
![add_ops-结构体成员函数](/assets/images/operators-2-cpp-add.png)  
可以看到，最后的call函数来到了c10::Dispatcher类  
![dispatcher-findop](/assets/images/dispatcher-findop.png) 
Dispatcher类的singleton函数会返回一个 static Dispatcher 类型的类成员，后续调用的两个成员函数都是这个静态成员的成员函数，这是设计模式中的单例模式。  
最后的 typed 函数返回的是一个 TypedOperatorHandle 类，它的 call 函数依旧是通过单例模式调用了静态成员的 call 函数  
![dispatcher-findop1](/assets/images/dispatcher-findop1.png)  
上述便是一个原生算子调用的整体过程，或者说是大致过程，因为略去了一些细节，这些细节涉及到 PyTorch 中一个名为 Dispatcher 的实现，而后续的算子注册过程也涉及到这个内容，所以下一篇根据 PyTorch 核心作者的演讲：
[Let’s talk about the PyTorch dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-PyTorch-dispatcher/)，结合代码学习下 dispatcher 的内部实现。
