---
title: 开源生图模型 stable diffusion lora 训练记录
categories: [模型训练]
description: 帮自媒体朋友做的生图模型训练研究
keywords: 
- stable diffusion
- 扩散模型
- lora 训练
date: 2023-08-16
draft: false
---

### 场景

- 自媒体运营，使用生成的穿搭图、风景图之类的图片引流

### 数据准备

- 穿搭图、旅游图：小红书下载穿搭、旅游博主的图片，本文示例图15张
- 抽象类设计图：设计师网站下载图片

### 训练流程

- 底模选择：sd 真人类型底模麦橘写实
- 训练脚本：kohya_ss

#### 数据预处理与参数

- 基本数据：训练图片15张图
- 数据增强：upscaler 到 2k，yolo 抠出人体，paddle matting 抠出更细致的人体轮廓，除去人体外的像素换成白底，左右翻转加强
- 训练硬件：显卡2080Ti
- 训练参数：network_alpha 32，network_dim 32，batchsize 8，重复次数6，训练轮次20轮

#### 结果分析

- 使用 xyz plot 插件查看训练轮次-lora强度的对比图：
- ![xyz plot](https://github.com/luokai0223/luokai0223.github.io/raw/master/pics/2023-08-16-stable_diffusion_lora/2.png)
- 本次测试的真人眼间距有一点点宽，中庭偏长，额头有点大。在 lora 强度较大时能明显发现五官缺陷被放大。
- 图片不宜过多，试过100多张图片，画面光影、背景、人物眼神异常，原始人物五官缺陷被放大。
- 训练轮次2轮内，强度设置0.7也能看见原始人物脸部特征，但是眼神和光影异常。
- 总结：15张图左右训练20轮次以内即可以完成，使用强度设置为0.5-0.7观感比较好，既有人物特点，也没有放大缺陷。
