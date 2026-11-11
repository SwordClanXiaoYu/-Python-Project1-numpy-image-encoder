# NumPy 实现 ViT 图像编码器

本项目使用 NumPy 完成 OpenCLIP ViT-B/32 图像编码器的前向计算。输入为经过预处理的 `[B, 3, 224, 224]` 图像数组，最终输出每张图像对应的 512 维单位向量。本人只实现前向计算，不包含模型训练。

## 实现内容

- 将图像划分为 49 个 `32 × 32` RGB 图像块；
- 完成多头注意力的拆头、注意力计算和多头合并；
- 完成 Transformer 块中的 LayerNorm、MLP 和两次残差连接；
- 加入 CLS token 和位置嵌入；
- 取最终 CLS token，完成输出投影和单位长度归一化。

核心实现位于 [`image_encoder.py`](image_encoder.py)。实验报告见 [`report/main.pdf`](report/实验报告.pdf)。

## 个人完成部分

本人完成 `image_encoder.py` 中 TODO-01 至 TODO-06 的核心实现，包括图像块重排、多头拆分、注意力计算、Transformer 残差与 MLP、token 嵌入以及最终 CLS 输出；并完成分阶段数值测试、结果分析和实验报告。课程提供的权重、预处理、数值辅助函数及检查程序不属于个人实现。

## 运行方法

该实现需要放入课程提供的 `project1-numpy-student` 项目中运行，因为模型权重、测试图片、辅助函数和参考结果由课程附件提供。安装依赖后，在原项目目录执行：

```Shell
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

`check.py` 会将当前实现的各阶段结果与课程提供的独立 FP32 参考数组进行比较。参考数组只用于测试，不会作为编码器输入。

## 测试结果

全部数值检查通过：

| 阶段        | 输出形状              |  最大绝对误差 |  平均绝对误差 |
| ----------- | --------------------- | ------------: | ------------: |
| preprocess  | `(10, 3, 224, 224)` | `0.000e+00` | `0.000e+00` |
| patches     | `(2, 49, 3072)`     | `0.000e+00` | `0.000e+00` |
| tokens      | `(2, 50, 768)`      | `4.005e-05` | `1.289e-06` |
| block0      | `(2, 50, 768)`      | `7.063e-06` | `3.250e-07` |
| last_tokens | `(2, 50, 768)`      | `4.768e-05` | `2.696e-06` |
| output      | `(10, 512)`         | `5.439e-07` | `8.311e-08` |
| batch       | `(3, 512)`          | `4.619e-07` | `7.632e-08` |

内部阶段的绝对误差阈值为 `2e-4`，最终输出阶段为 `2e-5`。实际误差均低于对应阈值，其中最终输出的最大绝对误差约为阈值的 `1/37`。

## 仓库说明

仓库只保留个人实现、实验报告和测试截图。课程提供的大型权重、参考数组、测试图片和原始压缩包未上传。Overleaf 模板使用

[github.com/ChaoFan996/Experiment-Report-Template-SZU](https://github.com/ChaoFan996/Experiment-Report-Template-SZU)

Overleaf 链接

https://www.overleaf.com/project/6af41cb7cb8d4f7336c8e6e3/share#b01bd44ca19cc1d2cfd191f2c48b3f48e3302ec3e2b78553

## AI 辅助说明

本实验使用 GPT-6 Astra 辅助调试，并帮助理解深度学习和数值计算相关概念。核心实验、代码和测试由个人完成。
