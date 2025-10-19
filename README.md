# U-Net COVID-19 CT图像分割

基于PyTorch实现U-Net网络，完成COVID-19肺部CT图像感染区域的语义分割。感谢claude知道喵喵喵🐱～～～

## 项目简介

- **任务**: COVID-19肺部感染区域分割
- **模型**: U-Net (31M参数)
- **数据集**: 2729张CT图像
- **效果**: Dice 0.84, IoU 0.74



## 项目结构
```
unet/
├── configs/config.py  # 配置文件
├── models/unet.py     # U-Net模型
├── utils/
│   ├── dataset.py     # 数据加载
│   └── metrics.py     # 评估指标
├── train.py           # 训练脚本
├── test.py            # 测试脚本
└── predict.py         # 预测脚本
```

## 使用方法
```bash
# 测试流程
python test.py

# 训练模型
python train.py

# 预测结果
python predict.py
```

## U-Net核心

**网络结构**: 编码器-解码器 + 跳跃连接
```
输入 (256×256×1)
  ↓ 编码器下采样
64 → 128 → 256 → 512 → 1024
  ↓ 解码器上采样 + 跳跃连接
512 → 256 → 128 → 64
  ↓
输出 (256×256×2)
```

## 训练结果

| 指标 | 数值 |
|------|------|
| Dice | 0.8444 |
| IoU | 0.7370 |
| 像素准确率 | 0.9953 |

## 作者
kk266299 
日期: 2025-10-19