# 2022-自动化测试-警告识别

## 1.简介

本项目主要是基于置信学习技术，对apache项目中的警告数据集进行去噪。

本项目的工作主要分为两个阶段：第一个阶段，利用github上现有的项目

findbugs-violation (https://github.com/lxyeah/findbugs-violations) ，收集github上的apache项目的警告数据集。第二个阶段，利用cleanlab这一python包，对数据集进行去噪，并对模型进行训练。

## 2.数据集收集

## 3.基于置信学习去噪

