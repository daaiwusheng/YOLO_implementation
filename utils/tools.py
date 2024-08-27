
import numpy as np

def normalize_image(image):
    """
    将图像数据从 [0, 255] 归一化到 [0, 1]。
    参数：
        image (numpy.ndarray): 输入的图像数组，范围 [0, 255]
    返回：
        numpy.ndarray: 归一化后的图像数组，范围 [0, 1]
    """
    # 归一化到 [0, 1]
    normalized_image = image.astype(np.float32) / 255.0
    return normalized_image




