import torch.nn as nn
import torch
import numpy as np
import cv2

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=None, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.03, eps=1e-3)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    img = np.zeros((600, 600), 'uint8')

    # 创建一个示例图像
    image = np.zeros((448, 448, 3), dtype=np.uint8)

    # 定义矩形参数
    top_left = (50, 50)
    bottom_right = (200, 200)
    color = (0, 255, 0)  # 绿色
    thickness = 2

    # 绘制矩形
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # 显示图像
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()