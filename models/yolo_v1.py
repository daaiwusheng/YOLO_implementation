import torch
import torch.nn as nn
from torchinfo import summary


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
# torch.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
# torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

class BackBone(nn.Module):
    def __init__(self, number_classes=20, initial_weights=True):
        super(BackBone, self).__init__()
        self.feature_calculator = nn.Sequential(
            # first layer
            nn.Conv2d(3, 64, 7, 2, 3),  # output width=224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # output width=112
            # second layer
            nn.Conv2d(64, 192, 3, padding=1),  # out put width=112
            nn.BatchNorm2d(192),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),  # output width=56
            # third layer
            # 3.1 sublayer
            nn.Conv2d(192, 128, 1),  # output width=56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 3.2 sublayer
            nn.Conv2d(128, 256, 3, padding=1),  # output width=56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 3.3 sublayer
            nn.Conv2d(256, 256, 1),  # output width=56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 3.4 sublayer
            nn.Conv2d(256, 512, 3, padding=1),  # output width=56
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),  # output width=28

            # 4th layer
            # 4.1 sublayer
            # 4.1.1 sublayer
            nn.Conv2d(512, 256, 1),  # output width=28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 4.1.2 sublayer
            nn.Conv2d(256, 512, 3, padding=1),  # output width=28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 4.2 sublayer
            # 4.2.1 sublayer
            nn.Conv2d(512, 256, 1),  # output width=28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 4.2.2 sublayer
            nn.Conv2d(256, 512, 3, padding=1),  # output width=28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 4.3 sublayer
            # 4.3.1 sublayer
            nn.Conv2d(512, 256, 1),  # output width=28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 4.3.2 sublayer
            nn.Conv2d(256, 512, 3, padding=1),  # output width=28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 4.4 sublayer
            # 4.4.1 sublayer
            nn.Conv2d(512, 256, 1),  # output width=28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # 4.4.2 sublayer
            nn.Conv2d(256, 512, 3, padding=1),  # output width=28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # 4.5 sublayer
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 4.6 sublayer
            nn.Conv2d(512, 1024, 3, padding=1),  # output width=28
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            nn.MaxPool2d(2),  # output width 14

            # 5th layer
            # 5.1 sublayer
            # 5.1.1 sublayer
            nn.Conv2d(1024, 512, 1),  # output width=14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 5.1.2 sublayer
            nn.Conv2d(512, 1024, 3, padding=1),  # output width=14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # 5.2.1 sublayer
            nn.Conv2d(1024, 512, 1),  # output width=14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # 5.2.2 sublayer
            nn.Conv2d(512, 1024, 3, padding=1),  # output width=14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # 5.3 sublayer
            nn.Conv2d(1024, 1024, 3, padding=1),  # output width=14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # 5.4 sublayer
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),  # output width=7
            # 6.1 sublayer
            nn.Conv2d(1024, 1024, 3, padding=1),  # output width=7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            # 6.2 sublayer
            nn.Conv2d(1024, 1024, 3, padding=1),  # output width=7, here the feature map is 7*7*1024 = 50176
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

        )

    def forward(self, x):
        return self.feature_calculator(x)


class DetectOriginal(nn.Module):
    def __init__(self, number_bbox=2, number_classes=20):
        super(DetectOriginal, self).__init__()
        self.detector = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 7 * 7 * (5 * number_bbox + number_classes))
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.detector(x)


class YoloVOne(nn.Module):
    def __init__(self, number_grid=7, number_bbox=2, number_classes=20):
        super(YoloVOne, self).__init__()
        self.number_grid = number_grid
        self.number_bbox = number_bbox
        self.number_classes = number_classes

        self.feature_extractor = BackBone(number_classes=self.number_classes)
        self.detector = DetectOriginal(number_bbox=self.number_bbox, number_classes=self.number_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.detector(x)
        x = x.view(-1, self.number_grid, self.number_grid, 5 * self.number_bbox + self.number_classes)
        return x


if __name__ == '__main__':
    yolo = YoloVOne()

    # Dummy image
    image = torch.randn(2, 3, 448, 448)  # torch.Size([2, 3, 448, 448])

    output = yolo(image)

    print(output.size())  # torch.Size([2, 7, 7, 30])
    # 打印模型总结
    summary(yolo, input_size=(1, 3, 448, 448))
