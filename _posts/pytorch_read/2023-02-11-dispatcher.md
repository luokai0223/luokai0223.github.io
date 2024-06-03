---
title: "PyTorch 源码阅读笔记（1）：dispatcher"
categories: [PyTorch 源码阅读笔记]
description: dispatcher
keywords: 
- PyTorch 源码阅读
tags: [PyTorch]
date: 2023-02-11
draft: false
---

## 什么是dispatcher
关于 PyTorch 的 dispatcher，PyTorch 的核心作者之一 Edward Z Yang 有过介绍：[Let’s talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-PyTorch-dispatcher/)  
PyTorch 作为多平台的神经网络框架，需要实现这样一种功能：每个通用的算子都要实现一些相同的 api，比如前传和反传，这些相同的api在不同的硬件设备会有不同的代码实现，CPU下可能要用到MKL，GPU下是CUDA，各个厂商的NPU加速卡也可能有不同的底层代码。PyTorch 需要根据不同的硬件设备和使用场景，调用对应的函数实现，dispatcher 能够实现这个功能。  
对于每个operator，dispatcher都会维护一个函数指针表，为每个dispatch key提供对应的实现。  
## Dispatcher
```cpp
class TORCH_API Dispatcher final {
    // 嵌套结构体
    struct OperatorDef final {
      explicit OperatorDef(OperatorName&& op_name)
      : op(std::move(op_name)) {}

      impl::OperatorEntry op;
      size_t def_count = 0;
      size_t def_and_impl_count = 0;
    };

  // 成员函数
    C10_ALWAYS_INLINE static Dispatcher& singleton() {
    // ...
    static Dispatcher& s = realSingleton();
    /*
    全局单例
    C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
      static Dispatcher _singleton;
      return _singleton;
     }
    */
    return s;
    }
    
  // 成员变量  
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  std::list<OperatorDef> operators_;
}
```  
operatorLookupTable_ 是一个算子表
LeftRight 实现参考：[Brief Announcement: Left-Right - A Concurrency Control Technique with Wait-Free Population Oblivious Reads](https://hal.archives-ouvertes.fr/hal-01207881/document)，大概逻辑是给任意的数据结构生成两份实例左和右，同时存在读写的时候，读左边的写右边的，写入完成后读取换到右边，当左边的所有读结束后，右边的写入再同步到左边，这种并发控制方式实现了零等待的读操作。  
flat_hash_map 实现参考：[A very fast hashtable](https://github.com/skarupke/flat_hash_map/blob/master/flat_hash_map.hpp)，是一种高效的哈希表。  
## DispatchKey 与 DispatchKeySet  
DispatchKey 是一个枚举类，不仅有针对不同后端（CPU、CUDA、XLA）的dispatch条目，也有像autograd和tracing这样的高抽象层级概念的条目。
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
dispatchkey 存储在 DispatchKeySet，DispatchKeySet 类有一个 uint64_t 类型的类成员 repr_，共计64个比特位，每个dispatch key 占用一个比特位：
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
DispatchKeySet里有多个key时，由于 dispatch key 的数字越大优先级越高，则比特位里面最左边的key优先级最高，每次执行都需要查找最高位的位置。  
最简单的方法自然是从左往右遍历，直到遇见比特位为1的值。而 PyTorch 用了 LLVM 项目提供的一种二分查找的方式计算最高位： 
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
这样循环的结果则是最高位前面的0的个数。相比遍历查找，时间复杂度由O(N)下降至O(logn)。  
## DispatchKeyExtractor  
```cpp
struct TORCH_API DispatchKeyExtractor final {

  void registerSchema(const FunctionSchema& schema) {
    TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }

private:
  static c10::utils::bitset makeBitsetForDispatchArgs(const FunctionSchema& schema) {
    TORCH_CHECK(schema.arguments().size() <= c10::utils::bitset::NUM_BITS(),
        "The function schema has ", schema.arguments().size(),
        " arguments but this PyTorch build only supports ", c10::utils::bitset::NUM_BITS());
    c10::utils::bitset dispatch_arg_indices_reverse;
    for (const auto index : c10::irange(schema.arguments().size())) {
      if (schema.arguments()[index].type()->isSubtypeOf(*TensorType::get()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *ListType::ofTensors()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *ListType::ofOptionalTensors()) ||
          schema.arguments()[index].type()->isSubtypeOf(
              *OptionalType::ofTensor())) {
        dispatch_arg_indices_reverse.set(schema.arguments().size() - 1 - index);
      }
    }
    return dispatch_arg_indices_reverse;
  }

  template<class... Args>
  DispatchKeySet getDispatchKeySetUnboxed(const Args&... args) const {
    auto ks = detail::multi_dispatch_key_set(args...);
    // Keys that are fallthrough should be skipped
    if (requiresBitsetPerBackend_) {
      auto backend_idx = ks.getBackendIndex();
      return impl::computeDispatchKeySet(ks, nonFallthroughKeysPerBackend_[backend_idx]);
    } else {
      return impl::computeDispatchKeySet(ks, nonFallthroughKeys_);
    }
  }

  c10::utils::bitset dispatch_arg_indices_reverse_;

// c10/util/Bitset.h
struct bitset final {
  using bitset_type = long long int;

  public:
    static constexpr size_t NUM_BITS() {
      return 8 * sizeof(bitset_type);
    }

  constexpr void set(size_t index) noexcept {
    bitset_ |= (static_cast<long long int>(1) << index);
  }
}
}
```
dispatch_arg_indices_reverse_ 是有64个比特位标记的结构体，上面代码根据算子 schema 的参数数量，在结构体内进行逆序（从右往左）的标记。  
getDispatchKeySetUnboxed 会根据算子的信息生成一个 DispatchKeySet，这个 DispatchKeySet 是排除掉了存储在本地线程中的某些 key。
例如当运行一个带有 autograd key 的算子时，会先构造反向计算图，构造完成后再运行前向计算的算子，这个时候就需要在第一次调用的时候对 autograd key 标记排除，才能真正调用算子：
```cpp
static inline DispatchKeySet computeDispatchKeySet(
    DispatchKeySet ks,
    // The key mask lets us eliminate (by zero entries) keys which should not
    // be considered for dispatch.  There are two cases when we use this:
    //
    // - If an operator's dispatch table contains a fallthrough entry, we
    //   should bypass it entirely when finding the key
    // - If a user invokes with redispatch, the mask lets us
    //   zero out the key the user asked us to stop.
    //
    // These excluded backends are NOT tracked in the TLS, but must be applied
    // AFTER TLS (since the backend may have been introduced for consideration
    // by the included TLS), which is why you have to pass them in to this
    // function (as opposed to just applying it to the input 'ks').
    DispatchKeySet key_mask
) {
  c10::impl::LocalDispatchKeySet local = c10::impl::tls_local_dispatch_key_set();
  // TODO: It's a bit irritating that we have to do logical ORs here, it would
  // be nice to only do one.  Can always_included be folded into the TLS?  Well,
  // it's a bit troublesome, because fastpath TLS access requires the type of
  // the TLS in question to be zero-initialized, so you don't actually win
  // anyting in that case.
  return (((ks | local.included_) - local.excluded_) & key_mask);
}
```