"""
图像预处理和数字分割工具
包含：二值化、形态学操作、数字分割、ROI 转 MNIST 格式
"""
import cv2
import numpy as np
import torch
from torchvision import transforms


def preprocess(img_bgr):
    """
    图像预处理：灰度化 -> 高斯模糊 -> 二值化 -> 形态学操作
    
    Args:
        img_bgr: BGR 格式的输入图像 (numpy array)
    
    Returns:
        binary: 二值图像，数字为白色(255)，背景为黑色(0)
    """
    # 1. 灰度化
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()
    
    # 2. 高斯模糊去噪，轻微模糊有助于二值化
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. 二值化：使用 Otsu 自适应阈值
    # Otsu 会自动选择最佳阈值，对手写数字效果较好
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 判断是否需要取反
    # 如果图像中白色像素更多，说明是白底黑字，需要取反
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    if white_pixels > total_pixels * 0.5:
        # 白底黑字，取反得到黑底白字
        binary = cv2.bitwise_not(binary)
    
    # 5. 形态学操作去噪
    # 开运算：去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算：连接断开的笔画
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary


def segment_digits(binary, min_area=100, min_height=20, max_aspect_ratio=8.0):
    """
    分割数字：使用轮廓检测提取每个数字的 bounding box
    
    Args:
        binary: 二值图像，数字为白色(255)，背景为黑色(0)
        min_area: 最小面积阈值，过滤小噪点 (默认: 100)
        min_height: 最小高度阈值，过滤太小的区域 (默认: 20)
        max_aspect_ratio: 最大宽高比，过滤过宽的区域 (默认: 8.0)
    
    Returns:
        rois: List[dict]，每个元素包含 {'roi': numpy array, 'bbox': (x, y, w, h)}
              按从左到右顺序排序
    """
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    img_height = binary.shape[0]
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # 过滤小噪点
        
        # 获取 bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤条件：
        # 1. bbox 高度不能太小
        if h < min_height:
            continue
        
        # 2. 宽高比不能太大（避免检测到横线等）
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > max_aspect_ratio:
            continue
        
        # 3. 位置过滤：数字应该在图像的主要区域，排除边缘的噪点
        # 如果高度太小，可能是噪点
        if h < img_height * 0.05:
            continue
        
        # 提取 ROI，稍微扩展一点边界，避免裁剪过紧
        margin = 3
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(binary.shape[1], x + w + margin)
        y_end = min(binary.shape[0], y + h + margin)
        
        roi = binary[y_start:y_end, x_start:x_end]
        
        if roi.size > 0:  # 确保 ROI 不为空
            rois.append({
                'roi': roi,
                'bbox': (x_start, y_start, x_end - x_start, y_end - y_start)
            })
    
    # 如果检测到的区域太多，可能是噪点太多，尝试更严格的过滤
    if len(rois) > 20:
        # 按面积排序，只保留最大的 N 个区域
        rois.sort(key=lambda r: r['roi'].shape[0] * r['roi'].shape[1], reverse=True)
        # 根据学号长度，保留前 10 个
        rois = rois[:10]
    
    # 按 y 坐标分组，找到主要的一行数字
    if len(rois) > 0:
        # 计算所有 ROI 的 y 中心点
        y_centers = [y + h // 2 for _, (x, y, w, h) in [(r['roi'], r['bbox']) for r in rois]]
        median_y = np.median(y_centers)
        
        # 只保留 y 中心点在 median_y 附近 ±30% 范围内的 ROI
        filtered_rois = []
        for roi_info in rois:
            x, y, w, h = roi_info['bbox']
            y_center = y + h // 2
            if abs(y_center - median_y) < img_height * 0.3:
                filtered_rois.append(roi_info)
        
        rois = filtered_rois if len(filtered_rois) > 0 else rois
    
    # 按 x 坐标从左到右排序
    rois.sort(key=lambda r: r['bbox'][0])
    
    return rois


def roi_to_mnist_tensor(roi, target_size=28, padding=4):
    """
    将 ROI 转换为 MNIST 格式的 tensor (1x28x28)
    
    处理流程：
    1. 保持长宽比缩放
    2. 在 28x28 画布上居中放置
    3. 归一化（与训练时一致）
    
    Args:
        roi: 二值 ROI 图像 (numpy array, 0-255)
        target_size: 目标尺寸 (默认: 28)
        padding: 边距像素数 (默认: 4)
    
    Returns:
        tensor: shape 为 (1, 28, 28) 的 float tensor，已归一化
    """
    h, w = roi.shape
    
    # 1. 计算缩放比例，保持长宽比，留出边距
    # 可用空间 = target_size - 2 * padding
    available_size = target_size - 2 * padding
    scale = min(available_size / h, available_size / w)
    
    # 确保缩放后至少有几个像素
    scale = max(scale, 0.1)
    
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    # 2. 缩放 ROI，使用 INTER_AREA 或 INTER_LINEAR
    if new_h > 0 and new_w > 0 and h > 0 and w > 0:
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = roi.copy()
        new_h, new_w = resized.shape
    
    # 3. 创建 28x28 的黑色画布
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 4. 计算居中位置
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # 确保不越界
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)
    
    # 5. 将缩放后的 ROI 放到画布中心
    y_end = min(y_offset + new_h, target_size)
    x_end = min(x_offset + new_w, target_size)
    
    # 确保索引有效
    src_h = min(new_h, y_end - y_offset)
    src_w = min(new_w, x_end - x_offset)
    
    if src_h > 0 and src_w > 0:
        canvas[y_offset:y_end, x_offset:x_end] = resized[:src_h, :src_w]
    
    # 6. 转换为 float 并归一化到 [0, 1]
    canvas_float = canvas.astype(np.float32) / 255.0
    
    # 7. 转换为 tensor 并添加 channel 维度
    tensor = torch.from_numpy(canvas_float).unsqueeze(0)  # (1, 28, 28)
    
    # 8. 应用与训练时一致的归一化（MNIST 的均值和标准差）
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    tensor = normalize(tensor)
    
    return tensor

