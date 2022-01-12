---
layout: post
title: pytorch源码阅读笔记（2）：operators算子——native算子注册
categories: [pytorch源码]
description: 看了一部分pytorch源码，总结记录一下
keywords: pytorch，libtorch
---

——————  

## 算子定义
首先运行如下宏：
```cpp
// TORCH_LIBRARY(aten, m) 
// 为什么要多写一句static void TORCH_LIBRARY_init_aten(torch::Library&); ？
// 宏展开后：
static void TORCH_LIBRARY_init_aten(torch::Library&); 
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_aten( torch::Library::DEF, &TORCH_LIBRARY_init_aten, "aten", c10::nullopt, __FILE__, __LINE__); 
void TORCH_LIBRARY_init_aten(torch::Library& m)
{
// ...
 m.def("add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
// ...
```
其中声明TORCH_LIBRARY_static_init_aten时，TORCH_LIBRARY_static_init_aten构造函数会在初始化成员变量Library lib_后运行TORCH_LIBRARY_init_aten(lib_)，执行函数内的一大串m.def(...)。
```cpp
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      c10::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};
```
Library 类内有多个def函数模板，这里会按如下方式调用：
```cpp
// ...
  template <typename Schema>
  Library& def(Schema&& raw_schema) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema));
    return _def(std::move(s));
  }

// ...
  Library& _def(
      c10::FunctionSchema&& schema,
      c10::OperatorName* out_name = nullptr) &;

// ...
#define DEF_PRELUDE "def(\"", schema.operator_name(), "\"): "
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name) & {
  TORCH_CHECK(kind_ == DEF || kind_ == FRAGMENT,
    DEF_PRELUDE,
    "Cannot define an operator inside of a ", toString(kind_), " block.  "
    "All def()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  ",
    ERROR_CONTEXT
  );
  TORCH_INTERNAL_ASSERT(ns_.has_value(), ERROR_CONTEXT);
  TORCH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
  auto ns_opt = schema.getNamespace();
  if (ns_opt.has_value()) {
    TORCH_CHECK(*ns_opt == *ns_,
      "Explicitly provided namespace (", *ns_opt, ") in schema string "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the (unique) TORCH_LIBRARY block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    bool b = schema.setNamespaceIfNotSet(ns_->c_str());
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  if (out_name) {
    *out_name = schema.operator_name(); // copy!
  }
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerDef(
      std::move(schema),
      debugString(file_, line_)
    )
  );
  return *this;
}
#undef DEF_PRELUDE
// ...
```
然后调用Dispatcher单例执行registerDef函数：
```cpp
// ...
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);
// ...
}
```
findOrRegisterName_函数首先调用findOp函数，在该函数内，调用Dispatcher单例的成员变量operatorLookupTable_的read方法：
```cpp
// ...
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

// ...
c10::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> c10::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  });
}

```
operatorLookupTable_的定义为：
```cpp
#if !defined(C10_MOBILE)
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
#else
  RWSafeLeftRightWrapper<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
```
LeftRight的大概逻辑是给任意的数据结构生成两份实例左和右，同时存在读写的时候，读左边的写右边的，写入完成后读取换到右边，当左边的所有读结束后，右边的写入再同步到左边，这种并发控制方式实现了零等待的读操作。  
read传入的lambda函数在flat_hash_map哈希表中搜索算子名称，搜索不到返回空指针，回到findOrRegisterName_函数，调用operatorLookupTable_的write函数写入OperatorName和OperatorHandle的键值对，findOrRegisterName_函数最终返回一个OperatorHandle，继续执行registerDef：
```cpp
RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name] {
    deregisterDef_(op, op_name);
  });
}
```
OperatorHandle op的成员变量operatorDef_定义为Dispatcher::OperatorDef*，它有成员变量impl::OperatorEntry op，执行registerSchema函数：
```cpp
// ...
void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (const auto& kernel : kernels_) {
    for (const auto &j : kernel.second) {
      if (j.inferred_function_schema != nullptr) {
        checkSchema(name_, schema, debug, *j.inferred_function_schema, j.debug);
      }
    }
  }
  // NB: don't register schema until after we've checked everything!
  // DispatchKeyExtractor dispatchKeyExtractor_;
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
}
```
impl::OperatorEntry op有成员变量执行DispatchKeyExtractor dispatchKeyExtractor_，执行dispatchKeyExtractor_的registerSchema函数：
```cpp
// ...
  void registerSchema(const FunctionSchema& schema) {
    TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }
// ...
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
```
更新dispatchKeyExtractor_的成员变量dispatch_arg_indices_reverse_，schema的成员arguments_类型为std::vector<Argument>，在makeBitsetForDispatchArgs函数内，遍历arguments_，符合类型判断后，传入逆序顺序值作为参数，调用set函数：
```cpp
struct TORCH_API DispatchKeyExtractor final {
// ...
  void registerSchema(const FunctionSchema& schema) {
    TORCH_INTERNAL_ASSERT(dispatch_arg_indices_reverse_.is_entirely_unset());
    dispatch_arg_indices_reverse_ = makeBitsetForDispatchArgs(schema);
  }
//...
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
// ...
c10::utils::bitset dispatch_arg_indices_reverse_;
// ...
```
dispatch_arg_indices_reverse的set函数，bitset_初始为0，每次生成一个1进行左移index位的运算，然后与bitset_进行按位或更新，这样每个符合类型的参数的逆序位置的bit值都为1，目的是"This avoids having to iterate over the stack to find all the tensors":
```cpp
// ...
  constexpr void set(size_t index) noexcept {
    bitset_ |= (static_cast<long long int>(1) << index);
  }
// ...  
```
registerDef最后返回一个RegistrationHandleRAII类，全局的Dispatcher单例和static常量TORCH_LIBRARY_static_init_aten都得到更新
## 算子实现
上述步骤根据schema调用def函数进行了算子定义，然后再运行如下宏：
```cpp
// TORCH_LIBRARY_IMPL(aten, CPU, m)
// 宏展开后：
static void C10_CONCATENATE( TORCH_LIBRARY_IMPL_init_aten_CPU_, 5)(torch::Library&);
static const torch::detail::TorchLibraryInit C10_CONCATENATE( TORCH_LIBRARY_IMPL_static_init_aten_CPU_, 5)( torch::Library::IMPL, c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check( c10::DispatchKey::CPU)>( []() { return &C10_CONCATENATE( TORCH_LIBRARY_IMPL_init_aten_CPU_, 5); }, []() { return [](torch::Library&) -> void {}; }), "aten", c10::make_optional(c10::DispatchKey::CPU), __FILE__, __LINE__);
void C10_CONCATENATE( TORCH_LIBRARY_IMPL_init_aten_CPU_, 5)(torch::Library & m)
{
    // ...
    m.impl("add.out", TORCH_FN(wrapper_add_out_out));
    // ...
}
```
与第一个宏的方式类似，同样定义一个static const TorchLibraryInit类，调用唯一的一个构造函数，运行一大串
m.impl，调用如下impl模板函数，
```cpp
// ...
  template <typename Name, typename Func>
  Library& impl(Name name, Func&& raw_f) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def
#if defined C10_MOBILE
    CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
#else
    CppFunction f(std::forward<Func>(raw_f));
#endif
    return _impl(name, std::move(f));
  }
// ...
```
这一步把上一步的函数指针经过转换，定义一个CppFunction类：
```cpp
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
```
然后调用_impl函数：
```cpp
#define IMPL_PRELUDE "impl(\"", name_str, "\", ...): "
Library& Library::_impl(const char* name_str, CppFunction&& f) & {
  auto name = torch::jit::parseName(name_str);
  auto ns_opt = name.getNamespace();
  // This is kind of similar to the checking in def(), but the error
  // messages are a little different for this call site
  if (ns_opt.has_value()) {
    // See Note [Redundancy in registration code is OK]
    TORCH_CHECK(*ns_opt == *ns_,
      IMPL_PRELUDE,
      "Explicitly provided namespace (", *ns_opt, ") in operator name "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the ", toString(kind_), " block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    bool b = name.setNamespaceIfNotSet(ns_->c_str());
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
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
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      // NOLINTNEXTLINE(performance-move-const-arg)
      std::move(f.cpp_signature_),
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}
#undef IMPL_PRELUDE
```
_impl函数内，经过一些check步骤，调用Dispatcher单例的registerImpl函数
```cpp
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    // NOLINTNEXTLINE(performance-move-const-arg)
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name, dispatch_key, handle] {
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}
```
findOrRegisterName_(op_name)会得到之前def进去的函数指针，下一步执行OperatorEntry::registerKernel函数：
```cpp
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
// check...

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  // Redirect catchAll registrations to CompositeImplicitAutograd.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  if (k[0].kernel.isValid()) {
#else
  if (k.size() > 0) {
#endif
    TORCH_WARN("Overriding a previously registered kernel for the same operator and the same dispatch key\n",
               "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
               "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
               "  dispatch key: ", toString(dispatch_key), "\n",
               "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : "no debug info"), "\n",
               "       new kernel: ", debug
    );
  }

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
// 其他函数定义...
```
auto& k...这一步中的kernels_是OperatorEntry的成员变量，类型与场景有关，这里当成是如下flat_hash_map表,存储了一个算子在相应DispatchKey下的函数表，按注册时间逆序排列，调用的时候取最新的:
```cpp
  ska::flat_hash_map<DispatchKey, std::list<AnnotatedKernel>>
```
定义k后的步骤则会先把对应的函数指针通过变量k存入哈希表，然后调用updateDispatchTable_函数：
```cpp
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
  // See Note [Undefined in dispatchTable_]
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  // Registration to CompositeExplicitAutograd and CompositeImplicitAutograd should be populated to Undefined.
  // We cannot do this above since Undefined cannot be represented in DispatchKeySet.
  if (dispatch_key == DispatchKey::CompositeImplicitAutograd || dispatch_key == DispatchKey::CompositeExplicitAutograd) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  // Note [Refresh Runtime Autograd entries in dispatchTable_]
  // Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
  if (c10::isBackendDispatchKey(dispatch_key)) {
    DispatchKey autograd_key = getAutogradKeyFromBackend(dispatch_key);
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}
```
updateDispatchTable_函数层层调用，会通过 auto kern_it = kernels_.find(dispatch_key) 获取核函数，赋值给dispatchTable_对应的位置。  
完成算子注册。