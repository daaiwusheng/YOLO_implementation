import torch.nn as nn
import torch


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
    # 假设输入的特征图形状为 (batch_size, channels, height, width)
    feature_map = torch.randn(3, 1024, 7, 7)  # 这里 batch_size = 1

    # 使用 view 方法展平
    flattened = feature_map.view(feature_map.size(0), -1)

    print(flattened.shape)  # 输出形状: torch.Size([1, 50176])

    # 假设输入的特征图形状为 (batch_size, channels, height, width)
    feature_map = torch.randn(1, 1024, 7, 7)

    # 使用 flatten 方法展平
    flattened = feature_map.flatten(start_dim=1)

    print(flattened.shape)  # 输出形状: torch.Size([1, 50176])
