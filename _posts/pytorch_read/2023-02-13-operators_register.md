---
title: "PyTorch 源码阅读笔记（2）：原生算子注册"
categories: [PyTorch 源码阅读笔记]
description: 原生算子注册
keywords: 
- PyTorch 源码阅读
- TorchScript
tags: [PyTorch]
date: 2023-02-13
draft: false
---

## 算子定义
按照官方描述，所有的原生算子（函数）都定义在aten/src/ATen/native/native_functions.yaml文件里面，以一个add算子为例：  
如下原生算子：  
```xml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: add.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: add_sparse
    SparseCsrCPU, SparseCsrCUDA: add_sparse_csr
    MkldnnCPU: mkldnn_add
    ZeroTensor: add_zerotensor
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_add_Tensor
  tags: [canonical, pointwise]
```
## 算子信息注册
算子通过如下宏进行 schema 注册：
```cpp
// 文件自动生成在 cmake-build-debug-wsl-gcc/aten/src/ATen/RegisterSchema.cpp
// TORCH_LIBRARY(aten, m) 展开如下
static void TORCH_LIBRARY_init_aten(torch::Library&);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_aten(
    torch::Library::DEF,
    &TORCH_LIBRARY_init_aten,
    "aten",
    c10::nullopt,
    "_file_name_",
    6);
void TORCH_LIBRARY_init_aten(torch::Library& m)
{
 m.def("add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", {at::Tag::core, at::Tag::pointwise});
}
```  
注册发生在 m.def(...)：  
```cpp
  template <typename Schema>
  Library& def(Schema&& raw_schema, const std::vector<at::Tag>& tags = {}, _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema));
    return _def(std::move(s), nullptr, tags, rv);
  }
```  
首先从 "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor" 生成一个 c10::FunctionSchema 对象实例：
```cpp
struct TORCH_API FunctionSchema {
//...
 private:
  OperatorName name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  bool is_vararg_;
  bool is_varret_;
}
```  
然后调用 _def(...)，关键步骤在:  
```cpp
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerDef(
          std::move(schema),
          debugString(file_, line_),
          tags
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForDef(schema);
      break;
  }
```  
上面代码中的c10::Dispatcher::singleton()会返回 Dispatcher 单例对象，然后调用 registerDef：
```cpp
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([this, op, op_name] {
    deregisterDef_(op, op_name);
  });
}
```
### name 注册
findOrRegisterName_ 函数：
```cpp
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != c10::nullopt) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  return handle;
}
```
算子还未注册，findOp 返回空指针，在 operators_.emplace_back(OperatorName(op_name)) 这一步，隐式构造了一个 OperatorDef 对象：
```cpp
class TORCH_API Dispatcher final {
    // 嵌套结构体
    struct OperatorDef final {
      explicit OperatorDef(OperatorName&& op_name)
      : op(std::move(op_name)) {}

      impl::OperatorDef op;
      size_t def_count = 0;
      size_t def_and_impl_count = 0;
    };
}
```  
OperatorDef 内又隐式构造了一个 类成员op，op 保存了算子的信息：  
```cpp
OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))
, schema_()
#ifndef C10_MOBILE
, tags_()
#endif
, dispatchTable_()
, dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
, kernels_()
, cpp_signature_()
, sym_cpp_signature_()
, is_observed_(ObservedOperators::isObserved(name_))
{
  // Pick up any backend fallbacks that were registered prior to this
  // OperatorEntry being created
  updateDispatchTableFull_(c10::Dispatcher::singleton());
}

class TORCH_API OperatorEntry final {
public:
  explicit OperatorEntry(OperatorName&& operator_name);
private:
  OperatorName name_;
  c10::optional<AnnotatedSchema> schema_;
  #ifndef C10_MOBILE
    std::vector<at::Tag> tags_;
  #endif
  std::array<KernelFunction, c10::num_runtime_entries> dispatchTable_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  // Pointer to the torch.ops.ns.op.overload object for speed
  c10::PyHandleCache py_cache_;

  ska::flat_hash_map<DispatchKey,
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
        // On mobile, we needn't worry about Jupyter notebooks.
        std::array<AnnotatedKernel, 1>
#else
        std::list<AnnotatedKernel>
#endif
        > kernels_;
}
```  
kernels_ 存储的是 DispatchKey 与对应 key 下面注册的核函数哈希表：  
```cpp
// This data structure represents a kernel that was registered to us from a
// user.  Unlike KernelFunction, AnnotatedKernel contains some extra metadata
// about the kernel that isn't necessary for actual dispatching (this is why
// we don't put AnnotatedKernel in the actual DispatchTable), but is useful for
// giving good error messages.
struct AnnotatedKernel final {
  AnnotatedKernel(KernelFunction k, std::unique_ptr<FunctionSchema> s, std::string d)
    : kernel(std::move(k))
    , inferred_function_schema(std::move(s))
    , debug(std::move(d))
    {}
  AnnotatedKernel() = default;
  KernelFunction kernel;
  std::unique_ptr<FunctionSchema> inferred_function_schema;
  // A little debug string to help us identify the kernel in question.
  // Most importantly it records the TORCH_LIBRARY block that did the
  // registration.
  std::string debug;
};
```  
然后 OperatorHandle handle(--operators_.end()) 构造了一个 OperatorHandle 对象：  
```cpp
private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorDef_(&*operatorIterator), operatorIterator_(operatorIterator)  {}
  friend class Dispatcher;
  template<class> friend class TypedOperatorHandle;
  // 当前算子信息
  Dispatcher::OperatorDef* operatorDef_;
  // 全局算子列表迭代器
  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
```
最后往全局单例 dispatcher 的成员变量 operatorLookupTable_ 写入 name - handle 对    
### schema 注册  
```cpp
void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug, std::vector<at::Tag> tags) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (const auto& kernel : kernels_) {
    for (const auto &j : kernel.second) {
      if (j.inferred_function_schema != nullptr) {
        checkSchema(name_, schema, debug, j.kernel, *j.inferred_function_schema, j.debug);
      }
    }
  }
  // NB: don't register schema until after we've checked everything!
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
  #ifndef C10_MOBILE
    tags_ = std::move(tags);
  #endif
}
```  
registerSchema 首先遍历 kernels_ , 对 AnnotatedKernel 进行检查；  
然后调用 dispatchKeyExtractor_.registerSchema(schema)（[参考 dispatcher](../dispatcher#Dispatchkeyextractor))记录参数信息；  
最后生成成员变量 schema_ 。 

## 算子函数注册
完成算子信息注册后，对于每个算子在对应的平台的实现，会调用下面的宏：
```cpp
// 对应代码是自动生成的，路径为 cmake-build-debug-wsl-gcc/aten/src/ATen/RegisterXXX.cpp
// TORCH_LIBRARY_IMPL(aten, CPU, m)，展开如下
static void TORCH_LIBRARY_IMPL_init_aten_CPU_1(torch::Library&);
static const torch::detail::
    TorchLibraryInit TORCH_LIBRARY_IMPL_static_init_aten_CPU_1(
        torch::Library::IMPL,
        c10::guts::if_constexpr<
            c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::CPU)>(
            []() { return &TORCH_LIBRARY_IMPL_init_aten_CPU_1; },
            []() { return [](torch::Library&) -> void {}; }),
        "aten",
        c10::make_optional(c10::DispatchKey::CPU),
        "_file_name_",
        31034);
void TORCH_LIBRARY_IMPL_init_aten_CPU_1(torch::Library& m)
{
    // ...
    m.impl("add.Tensor", TORCH_FN(wrapper_CPU_add_Tensor));
    // ...
}
```
与第一个宏的方式类似，impl 函数第一个参数是算子名称，第二个参数是函数指针：  
```cpp
TORCH_FN(wrapper_CPU_add_Tensor);
// 展开为
::c10::CompileTimeFunctionPointer<
    std::remove_pointer_t<
        std::remove_reference_t<decltype(wrapper_CPU_add_Tensor)>>,
    wrapper_CPU_add_Tensor>()
```
impl 函数
```cpp
  template <typename Name, typename Func>
  Library& impl(Name name, Func&& raw_f, _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def
#if defined C10_MOBILE
    CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
#else
    CppFunction f(std::forward<Func>(raw_f));
#endif
    return _impl(name, std::move(f), rv);
  }
```
函数内部实例化了一个 CppFunction 类型的变量：
```cpp
class TORCH_API CppFunction final {
  /// This overload accepts compile time function pointers, e.g.,
  /// `CppFunction(TORCH_FN(add_impl))`
  template <typename FuncPtr>
  explicit CppFunction(
      FuncPtr f,
      std::enable_if_t<
          c10::is_compile_time_function_pointer<FuncPtr>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedFunction(f)),
        cpp_signature_(
            c10::impl::CppSignature::make<typename FuncPtr::FuncType>()),
        schema_(c10::detail::inferFunctionSchemaFromFunctor<
                typename FuncPtr::FuncType>()),
        debug_() {}
 private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  c10::optional<c10::impl::CppSignature> cpp_signature_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;
}
```
然后对算子函数进行注册：
```cpp
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  // See Note [Redundancy in registration code is OK]
  TORCH_CHECK(!(f.dispatch_key_.has_value() &&
                dispatch_key_.has_value() &&
                *f.dispatch_key_ != *dispatch_key_),
    IMPL_PRELUDE,
    "Explicitly provided dispatch key (", *f.dispatch_key_, ") is inconsistent "
    "with the dispatch key of the enclosing ", toString(kind_), " block (", *dispatch_key_, ").  "
    "Please declare a separate ", toString(kind_), " block for this dispatch key and "
    "move your impl() there.  "
    ERROR_CONTEXT
  );
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          std::move(f.cpp_signature_),
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  return *this;
}
```
函数内再次使用了 Dispatcher 类的全局单例，调用 registerImpl 方法，找到之前注册的 OperatorEntry 后，调用：
```cpp
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // 注册函数签名
  // ...

  // 注册函数指针
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];
  // ...
  #ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
    k[0].kernel = std::move(kernel);
    k[0].inferred_function_schema = std::move(inferred_function_schema);
    k[0].debug = std::move(debug);
  #else
    k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  #endif
    AnnotatedKernelContainerIterator inserted = k.begin();
    // update the dispatch table, i.e. re-establish the invariant
    // that the dispatch table points to the newest kernel
    if (dispatch_key.has_value()) {
      updateDispatchTable_(dispatcher, *dispatch_key);
    } else {
      updateDispatchTableFull_(dispatcher);
    }
    return inserted;
}
```
## 算子函数实现
TORCH_FN 移除了算子函数的引用和指针类型，统一声明成了CompileTimeFunctionPointer 类型的结构体参与后续的注册。宏内的函数定义是自动生成的：  
```cpp
struct structured_ufunc_add_CPU_functional final : public at::native::structured_ufunc_add_CPU {
    void set_output_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(*outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
        TensorOptions options, DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          namedinference::propagate_names(*outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_ufunc_add_CPU::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return *outputs_[output_idx];
    }
    std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
};
at::Tensor wrapper_CPU_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
structured_ufunc_add_CPU_functional op;
op.meta(self, other, alpha);
op.impl(self, other, alpha, *op.outputs_[0]);
return std::move(op.outputs_[0]).take();
}
```  
wrapper_CPU_add_Tensor 函数内变量 op 相关的声明和定义在多个地方：  
```cpp
// cmake-build-debug-wsl-gcc/aten/src/ATen/ops/add_meta.h
namespace at {
namespace meta {
struct TORCH_API structured_add_Tensor : public TensorIteratorBase {
    void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};}}
// aten/src/ATen/native/BinaryOps.cpp
void structured_add_Tensor::meta // 由 TORCH_META_FUNC2(add, Tensor) 展开
(
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
// cmake-build-debug-wsl-gcc/aten/src/ATen/ops/add_native.h
namespace at {
namespace native {
struct TORCH_API structured_ufunc_add_CPU : public at::meta::structured_add_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};}}
// cmake-build-debug-wsl-gcc/aten/src/ATen/UfuncCPU_add.cpp
void structured_ufunc_add_CPU::impl // 由 TORCH_IMPL_FUNC(ufunc_add_CPU) 展开
(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_stub(device_type(), *this, alpha);
}
```
最终调用的是 add_stub ，add_stub 结构体重载了()操作符
add_stub 的声明与定义如下：
```cpp
// aten/src/ATen/native/BinaryOps.h
using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
// DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub) 宏展开如下：
struct add_stub : DispatchStub<structured_binary_fn_alpha, add_stub> { 
  add_stub() = default; 
  add_stub(const add_stub&) = delete; 
  add_stub& operator=(const add_stub&) = delete; 
  }; 
extern TORCH_API struct add_stub add_stub
// aten/src/ATen/native/BinaryOps.cpp 中也定义了一个add_stub，但是可见性不同
// DEFINE_DISPATCH(add_stub); 宏展开
struct add_stub add_stub
```
add_stub 结构体继承自 DispatchStub 的一个偏特化模板：
```cpp
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
#ifdef HAVE_AVX512_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX512)
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      , reinterpret_cast<void*>(VSX)
#endif
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

  void set_cuda_dispatch_ptr(FnPtr fn_ptr) {
    impl.cuda_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  void set_hip_dispatch_ptr(FnPtr fn_ptr) {
    impl.hip_dispatch_ptr = reinterpret_cast<void*>(fn_ptr);
  }

  static FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static FnPtr AVX2;
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  static FnPtr VSX;
#endif
private:
  DispatchStubImpl impl;
};
```
运行 REGISTER_DISPATCH 宏，在最后展开的代码中给 DispatchStub 的全特化模板的静态成员变量赋予对应硬件下的函数指针：
```cpp
// aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void add_kernel(TensorIteratorBase& iter, const Scalar& alpha_scalar) {
  if (iter.dtype() == ScalarType::Bool) {
      using scalar_t = bool;
      auto alpha = alpha_scalar.to<scalar_t>();
      cpu_kernel(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a + alpha * b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "add_cpu/sub_cpu", [&]() {
      auto alpha = alpha_scalar.to<scalar_t>();
      auto alpha_vec = Vectorized<scalar_t>(alpha);
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a + alpha * b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ {
          return vec::fmadd(b, alpha_vec, a);
        });
      });
  }
}
REGISTER_DISPATCH(add_stub, &add_kernel);
// REGISTER_DISPATCH 宏在不同的平台下有不同的定义，cpu下：
#elif defined(CPU_CAPABILITY)
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#define REGISTER_NO_AVX512_DISPATCH(name, fn_type)                             \
  REGISTER_AVX512_DISPATCH(name, static_cast<fn_type>(nullptr))

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> decltype(fn) DispatchStub<decltype(fn), struct name>::arch = fn;
```
每个原生算子的实现代码都在 native 文件夹下，经由如上步骤生成了对应的函数指针包装，参与到算子注册过程。