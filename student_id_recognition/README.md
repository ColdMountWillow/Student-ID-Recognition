# 手写学号识别项目

使用 PyTorch 实现的手写数字识别系统，能够识别照片中的手写学号。

## 环境安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
student_id_recognition/
  README.md              # 项目说明
  requirements.txt       # 依赖包
  train_mnist.py         # 训练脚本
  infer_student_id.py    # 推理脚本
  model.py               # CNN 模型定义
  utils_image.py         # 图像预处理和分割工具
  weights/               # 模型权重保存目录（自动创建）
  samples/               # 测试图片目录
```

## 使用步骤

### 1. 训练模型

使用 MNIST 数据集训练手写数字分类模型：

```bash
python train_mnist.py
```

**可选参数：**

- `--epochs`: 训练轮数（默认: 5）
- `--batch_size`: 批次大小（默认: 64）
- `--lr`: 学习率（默认: 0.001）
- `--device`: 设备类型，`cpu` 或 `cuda`（默认: cpu）
- `--save_path`: 模型保存路径（默认: weights/mnist_cnn.pt）

**示例：**

```bash
# 使用 GPU 训练 10 个 epoch
python train_mnist.py --epochs 10 --device cuda

# 自定义批次大小和学习率
python train_mnist.py --batch_size 128 --lr 0.0005
```

训练完成后，模型权重会保存到 `weights/mnist_cnn.pt`。

### 2. 识别学号

使用训练好的模型识别手写学号照片：

```bash
python infer_student_id.py --image <图片路径>
```

**参数：**

- `--image`: **必需**，输入图片路径
- `--weights`: 模型权重路径（默认: weights/mnist_cnn.pt）
- `--device`: 设备类型，`cpu` 或 `cuda`（默认: cpu）
- `--save_vis`: 可视化结果保存路径（例如: outputs/vis.jpg）

**示例：**

```bash
# 基本使用
python infer_student_id.py --image 2023217542.jpg

# 保存可视化结果
python infer_student_id.py --image 2023217542.jpg --save_vis outputs/vis.jpg

# 使用 GPU 推理
python infer_student_id.py --image 2023217542.jpg --device cuda
```

## 算法流程

### 训练流程

1. 使用 `torchvision` 下载 MNIST 数据集
2. 数据预处理：转换为 Tensor 并归一化（均值 0.1307，标准差 0.3081）
3. 训练 CNN 模型（3 层卷积 + 全连接层）
4. 在测试集上评估准确率
5. 保存模型权重

### 推理流程

1. **图像预处理** (`preprocess`):

   - 灰度化
   - 高斯模糊去噪
   - Otsu 自适应二值化
   - 判断并取反（确保数字为白色，背景为黑色）
   - 形态学操作去噪

2. **数字分割** (`segment_digits`):

   - 使用轮廓检测提取每个数字的 bounding box
   - 过滤小噪点（面积阈值）
   - 按 x 坐标从左到右排序

3. **ROI 转换** (`roi_to_mnist_tensor`):

   - 保持长宽比缩放 ROI
   - 在 28x28 画布上居中放置
   - 归一化（与训练时一致）

4. **模型预测**:

   - 逐个数字送入模型预测
   - 获取最大概率对应的类别（0-9）

5. **结果拼接**:
   - 按从左到右顺序拼接识别结果
   - 输出学号字符串
