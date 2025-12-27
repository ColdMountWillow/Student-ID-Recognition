"""
推理脚本：识别手写学号照片中的数字
输入：学号照片
输出：识别得到的学号字符串
"""
import argparse
import os
import cv2
import torch
from model import get_model
from utils_image import preprocess, segment_digits, roi_to_mnist_tensor


def infer_student_id(image_path, weights_path, device='cpu', save_vis=None):
    """
    识别学号照片中的数字
    
    Args:
        image_path: 输入图片路径
        weights_path: 模型权重路径
        device: 设备类型 ('cpu' 或 'cuda')
        save_vis: 可视化结果保存路径
    
    Returns:
        student_id: 识别得到的学号字符串
    """
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
    
    # 2. 加载模型
    print(f"加载模型: {weights_path}")
    model = get_model(num_classes=10, device=device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. 读取图片
    print(f"读取图片: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 4. 图像预处理：灰度化 -> 高斯模糊 -> 二值化 -> 形态学操作
    print("预处理图像...")
    binary = preprocess(img_bgr)
    
    # 5. 分割数字：使用轮廓检测提取每个数字的 bounding box
    print("分割数字...")
    rois = segment_digits(binary, min_area=100, min_height=20, max_aspect_ratio=8.0)
    
    if len(rois) == 0:
        raise ValueError("未检测到任何数字，请检查图片或调整分割参数")
    
    print(f"检测到 {len(rois)} 个数字区域")
    
    # 6. 逐个识别数字
    print("识别数字...")
    digits = []
    confidences = []
    device_tensor = torch.device(device)
    
    with torch.no_grad():
        for i, roi_info in enumerate(rois):
            roi = roi_info['roi']
            
            # 转换为 MNIST 格式：保持长宽比缩放 -> 居中放置 -> 归一化
            tensor = roi_to_mnist_tensor(roi)
            tensor = tensor.unsqueeze(0).to(device_tensor)  # (1, 1, 28, 28)
            
            # 模型预测
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            digit = predicted.item()
            conf = confidence.item()
            
            digits.append(str(digit))
            confidences.append(conf)
            
            print(f"  数字 {i+1}: {digit} (置信度: {conf:.3f})")
    
    # 7. 拼接学号字符串
    student_id = ''.join(digits)
    print(f"\n识别结果: {student_id}")
    
    # 8. 保存可视化结果
    if save_vis:
        os.makedirs(os.path.dirname(save_vis) if os.path.dirname(save_vis) else '.', exist_ok=True)
        
        vis_img = img_bgr.copy()
        for i, roi_info in enumerate(rois):
            x, y, w, h = roi_info['bbox']
            digit = digits[i]
            conf = confidences[i] if i < len(confidences) else 0.0
            
            # 根据置信度选择颜色：高置信度绿色，低置信度黄色
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
            
            # 画 bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # 标注数字和置信度
            label = f"{digit} ({conf:.2f})"
            cv2.putText(vis_img, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite(save_vis, vis_img)
        print(f"可视化结果已保存: {save_vis}")
    
    return student_id


def main():
    parser = argparse.ArgumentParser(description='识别手写学号照片中的数字')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('--weights', type=str, default='weights/mnist_cnn.pt',
                       help='模型权重路径 (默认: weights/mnist_cnn.pt)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='设备类型 (默认: cpu)')
    parser.add_argument('--save_vis', type=str, default=None,
                       help='可视化结果保存路径（例如: outputs/vis.jpg）')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = 'cpu'
    
    try:
        student_id = infer_student_id(
            image_path=args.image,
            weights_path=args.weights,
            device=args.device,
            save_vis=args.save_vis
        )
        print(f"\n最终识别结果: {student_id}")
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

