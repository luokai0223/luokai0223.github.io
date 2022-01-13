---
layout: post
title: PyTorch源码阅读笔记（4）：operators算子——native算子调用
categories: [PyTorch源码]
description: 看了一部分PyTorch源码，总结记录一下
keywords: PyTorch，libtorch
---

## 算子调用过程
算子在配置文件中被定义后，在编译过程中被代码生成脚本处读取，生成多处C++代码，以add.out算子为例，配置文件定义如下：
```xml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: add_out
    SparseCPU: add_out_sparse_cpu
    SparseCUDA: add_out_sparse_cuda
    SparseCsrCPU: add_out_sparse_csr_cpu
    SparseCsrCUDA: add_out_sparse_csr_cuda
    MkldnnCPU: mkldnn_add_out
```
当代码中使用此算子时，首先调用aten/src/ATen/ops/add.h 中定义的 add_out 内联函数：
```cpp
TORCH_API inline at::Tensor & add_out(at::Tensor & out, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha=1) {
    return at::_ops::add_out::call(self, other, alpha, out);
}
```
函数内继续调用 aten/src/ATen/ops/add_ops.h 以及 aten/src/ATen/Operators_*.cpp 中定义的 add_out 结构体的成员函数：
```cpp
struct TORCH_API add_out {
  using schema = at::Tensor & (const at::Tensor &, const at::Tensor &, const at::Scalar &, at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::add")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")
  static at::Tensor & call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);
  static at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out);
};
```
```cpp
// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<add_out::schema> create_add_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_out::name, add_out::overload_name)
      .typed<add_out::schema>();
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & add_out::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_add_out_typed_handle();
    return op.call(self, other, alpha, out);
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & add_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_add_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha, out);
}
```
可以看到，最后的call函数来到了c10::Dispatcher类:
```cpp
C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
  static Dispatcher _singleton;
  return _singleton;
}

c10::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> c10::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  });
}

c10::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  auto it = findOp(overload_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it;
    } else {
      return c10::nullopt;
    }
  } else {
    return it;
  }
}

OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    // Check if we have ANYTHING; if that's the case, that means you're
    // missing schema
    auto it2 = findOp({name, overload_name});
    if (!it2.has_value()) {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name);
    } else {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name,
        " but we found an implementation; did you forget to def() the operator?");
    }
  }
  return it.value();
}
```
最后的 typed 函数返回的是一个 TypedOperatorHandle 类，它的 call 函数依旧是通过单例模式调用了静态成员的 call 函数  
```cpp
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
public:
  // ...
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }
  // ...
};
```