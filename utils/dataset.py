import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import math


class VOCDataset(Dataset):
    def __init__(self, index_file_path, image_size=448, grid_size=7, number_bbox=2, number_classes=20):
        super(VOCDataset, self).__init__()
        self.to_tensor = transforms.ToTensor()
        self.index_file_path = index_file_path
        self.image_size = image_size
        self.grid_size = grid_size
        self.number_bbox = number_bbox
        self.number_classes = number_classes

        self.image_paths_list = []
        self.boxes_list = []
        self.labels_list = []

        count = 0
        with open(index_file_path, 'r') as f_data_index:
            for line in f_data_index:
                image_full_path = line.rstrip() + '.jpg'
                label_file_full_path = line.rstrip() + '.txt'
                self.image_paths_list.append(image_full_path)
                with open(label_file_full_path, 'r') as f_label_file:
                    boxes = []
                    labels = []
                    for line_label in f_label_file:
                        c, center_x, center_y, w, h = map(float, line_label.rstrip().split())
                        c = int(c)
                        boxes.append([center_x, center_y, w, h])
                        labels.append(c)
                    self.boxes_list.append(boxes)
                    self.labels_list.append(labels)
                f_label_file.close()
                count += 1
                print(line)
        f_data_index.close()
        print(
            f'{count}, images number is {len(self.image_paths_list)}, {len(self.boxes_list)},  {len(self.labels_list)}')

    def __len__(self):
        return len(self.image_paths_list)

    def encode_label(self, index):
        # encode the bbox for yolo to calculate loss value
        number_label_dimension = 4 + 1 + self.number_classes
        target_label = np.zeros([self.grid_size, self.grid_size, number_label_dimension], dtype=float)
        boxes_np = np.array(self.boxes_list[index])
        labels_np = np.array(self.labels_list[index])
        cell_size = 1.0 / self.grid_size  # cause bbox center coordinate is a factor to width and height
        for index_of_bbox, bbox in enumerate(boxes_np):
            center_xy = bbox[0:2]
            j = math.floor(center_xy[0]/cell_size)
            i = math.floor(center_xy[1]/cell_size)

            bbox_wh = bbox[2:4]
            label = labels_np[index_of_bbox]
            corner_left_top = np.array([i*cell_size, j*cell_size])
            real_center_xy = (center_xy - corner_left_top) / cell_size

            target_label[i, j, 0:2] = real_center_xy
            target_label[i, j, 2:4] = bbox_wh
            target_label[i, j, 4] = 1.0
            target_label[i, j, 5 + label] = 1.0
        return target_label

    def __getitem__(self, index):
        # encode label for calculating loss value
        target_label = self.encode_label(index)
        target_label = torch.as_tensor(target_label)
        # handle image
        current_image_path = self.image_paths_list[index]
        current_image = cv2.imread(current_image_path)
        current_image = cv2.resize(current_image, dsize=(self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LINEAR)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
        current_image = np.ascontiguousarray(current_image, dtype=np.float32)
        current_image = self.to_tensor(current_image)

        return current_image, target_label


def draw_bbox(image, bbox, grid_size, cell_size, image_size, color=(0, 255, 0), thickness=2):
    # bbox = [center_x, center_y, w, h] (相对坐标)
    center_xy = bbox[0:2]
    bbox_wh = bbox[2:4]

    # 计算边界框的格子位置
    cell_x = int(center_xy[0] * grid_size)
    cell_y = int(center_xy[1] * grid_size)

    # 计算边界框的实际坐标
    center_x, center_y = center_xy
    w, h = bbox_wh

    # 转换为实际像素坐标
    center_x_pixel = int(center_x * image_size)
    center_y_pixel = int(center_y * image_size)
    w_pixel = int(w * image_size)
    h_pixel = int(h * image_size)

    # 计算边界框的顶点坐标
    top_left = (int(center_x_pixel - w_pixel / 2), int(center_y_pixel - h_pixel / 2))
    bottom_right = (int(center_x_pixel + w_pixel / 2), int(center_y_pixel + h_pixel / 2))

    # 限制坐标在图像范围内
    top_left = (max(0, top_left[0]), max(0, top_left[1]))
    bottom_right = (min(image.shape[1] - 1, bottom_right[0]), min(image.shape[0] - 1, bottom_right[1]))

    # 绘制边界框
    # 确保 image 是 uint8 类型的连续数组
    if not isinstance(image, np.ndarray):
        raise TypeError("image should be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError("image dtype should be uint8")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("image should be a 3-channel (H, W, C) array")
    image = np.zeros((448, 448, 3), dtype=np.uint8)
    image = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image


# 测试代码
def test_voc_dataset():
    # 设置路径
    index_file_path = '/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex/train.txt'
    image_size = 448
    grid_size = 7
    number_bbox = 2
    number_classes = 20

    # 实例化数据集
    dataset = VOCDataset(index_file_path, image_size, grid_size, number_bbox, number_classes)

    # 测试数据集的长度
    print(f"Dataset length: {len(dataset)}")

    # 测试加载单个数据样本
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 获取一个批次的数据
    for images, labels in data_loader:
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")

        # 显示图像
        image_np = images[0].numpy().transpose(1, 2, 0)  # 转换为(H, W, C)
        image_np = (image_np * 255).astype(np.uint8)  # 转换为uint8类型

        # 画边界框
        for i in range(grid_size):
            for j in range(grid_size):
                if labels[0][i][j][4] == 1.0:  # 仅当有目标存在时
                    bbox = labels[0][i][j][0:4]  # 获取边界框坐标
                    # 计算当前格子位置
                    cell_size = 1.0 / grid_size
                    bbox_offset = [i * cell_size, j * cell_size]
                    bbox_center = [bbox[0] * cell_size + bbox_offset[0], bbox[1] * cell_size + bbox_offset[1]]
                    bbox_wh = [bbox[2] * cell_size, bbox[3] * cell_size]
                    bbox_in_image = [bbox_center[0], bbox_center[1], bbox_wh[0], bbox_wh[1]]

                    image_np = draw_bbox(image_np, bbox_in_image, grid_size, cell_size, image_size)

        # 使用OpenCV显示图片
        cv2.imshow('Image', image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # break  # 只测试一个batch


if __name__ == '__main__':
    # train_dataset = VOCDataset('/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex/train.txt')
    test_voc_dataset()
