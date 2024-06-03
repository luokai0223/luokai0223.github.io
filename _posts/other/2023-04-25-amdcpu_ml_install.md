---
title: 使用AMD显卡搭建深度学习环境
categories: [环境配置]
description: 记录A卡机器学习配置与使用
keywords: 
- AMD 机器学习
- AMD 深度学习
- AMD PyTorch
- Deep Learning
- ROCm
- DirectML
date: 2023-04-25
draft: false
---

### 总结

- 在 linux 系统下基于 ROCm 底层，基本可用，最大的影响是少部分算子性能有问题，其次是现在很多开源项目的量化方案完全基于 CUDA api，比如chatglm-6b，量化层必须在 CUDA 环境下使用。
- 在 windows 下基于 DirectML 底层，Pytorch 框架下涉及到 `**.cuda()`，`**.to('cuda')`等方式写的代码都需要修改 。tensorflow 则没有这个问题，不过现在开源模型用 tensorflow 比较少了。

### ROCm

机器学习/深度学习框架的加速实现一般都高度依赖于硬件对应的底层加速方案，英伟达的 CUDA，英特尔的 OpenVINO，各家 NPU 以及加速卡的闭源库，AMD 官方的底层加速框架是 ROCm, 首先尝试了此方案。

打开官网文档，查看 Prerequisite Actions 页面，很遗憾发现我的 6600XT 没在文档里写的确定支持的硬件列表内，不过感觉放弃太快有些不甘心，搜索引擎搜了些关键词，有成功有失败，研究一番后按下面步骤安装（ROCm 只支持linux，我使用了 Ubuntu 20.04，内核版本5.13.0-30-generic）：
#### 准备步骤
```shell
sudo apt update
sudo apt-get install wget gnupg2 
sudo usermod -a -G video $LOGNAME
sudo usermod -a -G render $LOGNAME
echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
echo 'EXTRA_GROUPS=render' | sudo tee -a /etc/adduser.conf
```
#### 使用官方安装脚本
我用的是 https://repo.radeon.com/amdgpu-install/22.10.1/ubuntu/focal/amdgpu-install_22.10.1.50101-1_all.deb。
不同的系统和rocm版本，对应的链接不一样，摸索了一下存档逻辑，所有脚本都在repo.radeon.com/amdgpu-install目录下，接着类似日期的数字22.10.1，越大 rocm 版本越高，然后的两级目录是发行版和发行版名称，最后文件名里面的50101表示5.1.1版本，安装完成后按如下命令运行脚本：
```shell
sudo amdgpu-install --usecase=dkms
amdgpu-install -y --usecase=rocm
```
安装过程会用到amd官方域名下的包，下载比较慢，软件后运行 rocm-smi, 看到如下结果：
```shell
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK  MCLK   Fan  Perf  PwrCap  VRAM%  GPU%  
0    36.0c  3.0W    0Mhz  96Mhz  0%   auto  130.0W    8%   0%    
================================================================================
============================= End of ROCm SMI Log ==============================

```
运行 /opt/rocm/bin/rocminfo 可以看到设备信息，我的显卡设备显示为：
```shell
  Name:                    gfx1032                            
  Uuid:                    GPU-XX                             
  Marketing Name:          AMD Radeon RX 6600 XT
```
看起来应该是安装成功了，下面测试一下深度学习框架。
#### 测试
##### PyTorch
按照 PyTorch 官网的 ROCm 版本安装命令安装，运行时如果有找不到共享库的报错，则增加环境变量 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib, 使用的时候，直接当成CUDA设备处理，出现下面的结果说明正常：
```python
In [13]: torch.cuda.is_available()
Out[13]: True
```
运行如果报 hipErrorNoBinaryForGpu: Unable to find code object for all current devices!, 说明存在设备兼容性的问题，我的 6600XT 通过增加环境变量 export HSA_OVERRIDE_GFX_VERSION=10.3.0 解决了，其他型号的显卡可能也可以修改这一项为对应的值解决。

然后试一下运行的时候是否真的调用显卡，而不是回落到cpu，运行下面测试：
```python
In [12]: while True:
    ...:     torch.randn((999, 999), device=torch.device("cuda:0"))**20
```
运行的时候查看 rocm-smi 命令的结果：
```shell
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK     MCLK     Fan     Perf  PwrCap  VRAM%  GPU%  
0    71.0c  126.0W  2375Mhz  1000Mhz  70.98%  auto  130.0W   11%   99%   
================================================================================
============================= End of ROCm SMI Log ==============================
```
确实是正常运行的，显卡风扇也开始狂转。
##### TensorFlow
运行 pip install tensorflow-rocm 下载tensorflow，安装完后运行，报了一些找不到共享库的错误，运行 sudo apt install rocm-libs hipcub miopen-hip，安装完依赖后，成功运行
```python
In [4]: tf.config.list_physical_devices('GPU')
Out[4]: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
##### benchmark
跑了一下 ai-benchmark，结果是：
```shell
Device Inference Score: 7149
Device Training Score: 5825
Device AI Score: 12974
```
大概1080 的水平，这个分数每个版本都有变化，曾经分数十分低，应该是 ROCm 加速库对某些算子的处理有问题，导致运行效率过低，以前用华为的昇腾加速卡也遇见过在 yolo 的 concat 层耗时特别长，用了算子补丁才恢复理想效果。
### DirectML
ROCm 只能运行于linux，因为平时避免不了使用 office 套件和一些只有 windows 有的软件，所以日常使用的是 window 11 + wsl2，wsl内可以直接调用 CUDA，达到很好的使用体验，找了下似乎没发现 ROCm 能够达到类似效果，但是发现微软出的加速后端 DirectML 也能支持 AMD 显卡进行机器学习，于是开始尝试。
#### 安装
DirectML 安装过程比较简单，选择需求按照官方一步步即可 https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro
##### PyTorch
官网上说想要在 wsl 里面使用，需要 windows 11, 我的台式机只有 windows 10, 于是就直接在 windows 中进行测试：

```shell
conda install numpy pandas tensorboard matplotlib tqdm pyyaml -y
pip install opencv-python
pip install wget
pip install torchvision

conda install pytorch cpuonly -c pytorch
pip install torch-directml
```

这个方式必须使用微软支持的 pytorch 版本(1.8 和 1.13)，同时代码上用法是：

```python
model.to("dml")
```
很明显带来两个问题：

第一个是写法差异导致的问题，很多项目的写法都是**.cuda()，这样已有代码涉及到调用显卡的地方都要修改；

第二是算子实现的滞后性，我测试了一个生成网络，在 1.8 版本下遇见 aten::reflection_pad2d 算子找不到，抄了一下新版的层实现代码，覆盖了模型的 reflection_pad2d 层后可以正常使用。

##### TensorFlow
tensorflow1 的包支持 1.15 版本，tensorflow2 的包支持 2.10，两者的安装和用法微软网站上也有

```shell
pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin
```

##### onnxruntime
onnxruntime 算是 DirectML 下使用最方便的框架了，跟随最新版，刚刚 pytorch 里的网络导出 onnx 跑没有遇见算子未实现的问题，加速效果也很好。
##### benchmark
在 DirectML 上跑分的结果是:
```shell
Device Inference Score: 8001
Device Training Score: 8872
Device AI Score: 16873
```
比 rocm 的分数要高