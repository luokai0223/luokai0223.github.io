---
layout: post
title: pytorch源码阅读笔记（2）：operators算子——native算子注册
categories: [pytorch源码]
description: 看了一部分pytorch源码，总结记录一下
keywords: pytorch，libtorch
---

最开始想要看看pytorch源码是从自定义算子开始，所以先看看pytorch的native算子（翻译似乎是原生算子？因为最终的具体实现都是和平台相关的，比如CUDA代码）是如何运行的  
——————  

## 算子基本信息定义
按照官方描述，所有的原生算子（函数）都定义在aten/src/ATen/native/native_functions.yaml文件里面，以一个add算子为例：  
![算子示例-add](/assets/images/native-functions-add.png)  
配置文件中定义了算子的输入输出、支持平台等信息，具体内容官方有详细描述。  
dispatch涉及到 pytorch 中一个名为  Dispatcher 的实现，具体细节为另一个话题，暂时先简单理解为一个带索引的数据结构。  
## 算子注册
算子在配置文件中被定义后，在编译过程中被代码生成脚本处读取，生成多处C++代码：
在aten/src/ATen/RegisterSchema.cpp文件中，

