---
title: 论文阅读笔记：遮挡环境下的面部识别概述
categories: [论文阅读]
description: 论文阅读
tags: [CNN, Deep Learning, Face Recognition, Multi-Branch ConvNets]
date: 2022-02-08
---

## 概述
面部遮挡的场景可能有：  
* 脸部配饰：眼镜、口罩、帽子、头发。
* 外部遮挡：被手或者各种其他物体遮挡。
* 人脸局部截取：在某些受限环境和视野下的局部面部捕捉。
* 人工遮挡：人工生成的局部色块、随机噪声等。


## 人脸检测
人脸检测是人脸识别流水线的第一步，人脸被大面积遮挡的时，类内相似度和类内偏差增加，人脸检测遇到挑战。许多方法通过自适应技术解决这个问题。
### 通用人脸检测
一般人脸检测算法为了应对遮挡情况，都有一些特殊的算法处理。MTCNN, Sckit-Image 和 Haar cascades 在实验设置或者室内环境下可以较好地检测到被遮挡的人脸。从方法来看 ，人脸检测技术主要可以分成三类：
* Rigid templates：Harr 类和 AdaBoost 均属于此类，在实时系统中性能较差。
* DPM(deformable part models): 实时系统表现有改善，但是计算复杂度较高。
* DCNN：DCNN provide a solid  solution for various A-PIE problems up to date。
### 遮挡下的人脸检测
处理遮挡人脸的方式主要分为以下三类：
* 定位面部可见区域：CNN 类提取人脸局部特征。
* 丢弃被遮挡子区域的特征：FAN, LLE-CNN, AdaBoost cascade classifier，计算复杂度低。
* 使用遮挡信息：DCNN 通过提取遮挡区域附近的特征，尝试减少遮挡的影响。


## 人脸识别
### 人脸识别流水线
1. Face Detection：检测图像中的人脸。
2. Face Processing：人脸裁剪、缩放、对齐等。
3. Feature Extraction：提取人脸特征。
4. Face Matching：从图像数据库中匹配最相似的特征向量。


##  occlusion-robust 人脸识别
特征提取时，未遮挡区域的特征鲁棒性高于遮挡区域，但是在实际场景中遮挡位置的不确定性可能产生问题。
遮挡鲁棒技术使用新的相似性计算方法和更少的层数来处理类内相似度。Learning-based 特征更适合先进的系统。
Learning-based 特征可以分为四类：
* appearance-based：使用眼睛附近的子空间进行判别学习。
* scoring-based：使用统计学习方法计算面部不同区域的遮挡概率，然后选择合适的区域进行判断。
* Sparse representation classifier：对遮挡区域进行划分和识别，然后利用掩码策略和字典学习技术进行重建。
* deep learning-based：训练成本高。

## 口罩对人脸识别的影响
口罩会对面部进行大面积的遮挡，导致了更高的类内相似度和类内偏差，对人脸识别的验证过程有较大影响。
在多种算法上测试的结果，遮挡面积约 70% 时，false rejection rates 为 20% 至 50%。

## 总结
对于戴口罩人脸识别，现在没有完善的解决方法，肯定会影响识别效果。iOS 最新的系统有了戴口罩识别功能，苹果的解决方式是识别眼部附近的特征，并且 iPhone 的结构光摄像头采集到的特征信息比一般的摄像头要多，才达到了支付级别的精度。