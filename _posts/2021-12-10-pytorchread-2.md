---
layout: post
title: PyTorch源码阅读笔记（2）：dispatcher
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch, dispatcher
---

关于 PyTorch 的 dispatcher ，PyTorch 的核心作者之一 Edward Z Yang 有过介绍：[Let’s talk about the PyTorch dispatcher](http://blog.ezyang.com/2020/09/lets-talk-about-the-PyTorch-dispatcher/)，本篇笔记内容是自己根据大佬的讲解进行学习的一些记录。  
——————  

## 什么是dispatcher
PyTorch 作为多平台的神经网络框架，需要实现这样一种功能：每个通用的算子都要实现一些相同的 api，比如前传和反传，这些相同的api在不同的硬件设备会有不同的代码实现，CPU下可能要用到MKL，GPU下是CUDA，各个厂商的NPU加速卡也可能有不同的底层代码。PyTorch 需要根据不同的硬件设备和使用场景，调用对应的函数实现，dispatcher 能够实现这个功能。  
对于每个operator，dispatcher都会维护一个函数指针表，为每个dispatch key提供对应的实现。  

## dispatcher key
DispatchKey 是一个unsigned char类型枚举类，不仅有针对不同后端（CPU、CUDA、XLA）的dispatch条目，也有像autograd和tracing这样的高抽象层级概念的条目。
```cpp
typedef unsigned char uint8_t
enum class DispatchKey : uint8_t {
  Undefined = 0,
  CatchAll = Undefined,
  CPU, // registered at build/aten/src/ATen/RegisterCPU.cpp
  CUDA, // registered at build/aten/src/ATen/RegisterCUDA.cpp
  HIP, // NB: I think this is not actually used, due to Note [Masquerading as
  FPGA, // Xilinx support lives out of tree at
       // ......
}
```
dispatch key 存储在 DispatchKeySet，DispatchKeySet 类有一个unsigned long类型的类成员repr_，repr_有64个比特位，每个dispatch key 占用一个比特位：
```cpp
// DispatchKeySet构造函数之一，传入key时，把对应比特位的数值标记为1
  explicit constexpr DispatchKeySet(DispatchKey t)
      : repr_(
            t == DispatchKey::Undefined
                ? 0
                : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
```
存储多个key时，直接进行按位或的操作：
```cpp
// 重载操作符
  constexpr DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }
// 新增key调用重载的或操作符
  C10_NODISCARD DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
```

## key查找
DispatchKeySet里有多个key时，由于 dispatch key 的数字越大优先级越高，则比特位里面最左边的key优先级最高，每次执行都需要查找最高位的位置。  
最简单的方法自然是从左往右遍历，直到遇见比特位为1的值。当然大佬们不可能这么简单，PyTorch 用了 LLVM 项目提供的一种二分查找的方式计算最高位：
```cpp
// c10/core/DispatchKeySet.h
  DispatchKey highestPriorityTypeId() const {
    return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
  }

// c10/util/llvmMathExtras.h
namespace detail {
template <typename T, std::size_t SizeOfT>
struct LeadingZerosCounter {
  static std::size_t count(T Val, ZeroBehavior) {
    if (!Val)
      return std::numeric_limits<T>::digits;

    // Bisection method.
    std::size_t ZeroBits = 0;
    for (T Shift = std::numeric_limits<T>::digits >> 1; Shift; Shift >>= 1) {
      T Tmp = Val >> Shift;
      if (Tmp)
        Val = Tmp;
      else
        ZeroBits |= Shift;
    }
    return ZeroBits;
  }
};
```
Val为传入的repr_，循环内的Shift值依次为32、16、8、4、2、1（二进制值分别为100000，10000，1000， 100，10，1），每次循环把Val右移Shift位：  
如果右移后的值不为0的话，说明还存在值为1的比特位，把右移后的值赋予Val，继续下一次右移；  
如果右移后的值为0的话，说明右移的位数范围内没有1，使用按位或保存至ZeroBits，继续下一次右移。  
这样循环的结果则是最高位前面的0的个数。相比遍历查找，时间复杂度由O(N
)下降至O(logn)。

以上便是 dispatcher 较基本的功能和实现，最开头那篇博文关于 dispatcher 实现的描述内容不止这些，不过由于 dispatcher 是一个“横切关注点”，涉及到很多模块，后续就从对应模块出发，涉及到了相关功能再补充。