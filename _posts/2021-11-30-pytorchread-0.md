---
layout: post
title: PyTorch源码阅读笔记：笔记目录
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---

最近断断续续看了部分PyTorch源代码，整理一下零散的笔记  
——————

*边看边更新，PyTorch项目庞大，预计看的方式整体为 C++到 Python，按照主要核心功能从整体到局部，整理笔记的时候 PyTorch 最新版本为1.10.1，按照此版本记录*  


### [1、代码结构与libtorch编译](https://luokai0223.github.io/2021/12/01/pytorchread-1/)
PyTorch 项目庞大，首先整体了解下代码的目录结构，然后从较底层的 C++ 代码开始阅读，libtorch 为 PyTorch 的独立 C++ 共享库，按照官方提供的 cmake 文件和配置参数进行编译。  
### [2、dispatcher](https://luokai0223.github.io/2021/12/10/pytorchread-2/)
Dispatcher 为 PyTorch 中的一个横切关注点，此篇分析一下 dispatch key 的存储逻辑以及查找逻辑，主要涉及到一个二分查找算法。
### [3、原生算子注册](https://luokai0223.github.io/2022/01/03/pytorchread-3/)
PyTorch 带有大量原生（native）算子，算子的注册与调用过程以 Dispatcher 为核心，涉及多种数据结构。在编程范式方面，用到了面向对象编程、宏编程、函数式编程，本篇主要理清这一过程的逻辑。
### [4、张量库](https://luokai0223.github.io/2022/01/03/pytorchread-4/)
PyTorch 实现了一个自动微分张量库，设计上对张量概念与底层存储进行解耦。代码层面也包含了大量自动生成的代码用以实现调用算子，另外还有一个比较底层的模块 —— C10 模块的侵入式指针，此篇主要对这几块进行分析。
### 5、TorchScript