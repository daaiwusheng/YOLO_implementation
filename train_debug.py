import torch
from models.yolo_v1 import YoloVOne
from torchinfo import summary
from utils.dataset import VOCDataset
from torch.utils.data import  DataLoader

print(f'CUDA Device Count: {torch.cuda.device_count()}')
device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
print(device)


def train():
    yolo_v_1 = YoloVOne(number_grid=7, number_bbox=2, number_classes=20)
    summary(yolo_v_1, input_size=(1, 3, 448, 448))
    # create optimizer
    optimizer = torch.optim.SGD(yolo_v_1.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    # load dataset
    train_index_file_path = '/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex/train.txt'
    val_index_file_path = '/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex/val.txt'
    image_size = 448
    grid_number = 7
    number_bbox = 2
    number_classes = 20
    # train data
    dataset_train = VOCDataset(train_index_file_path, image_size, grid_number, number_bbox, number_classes)
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)
    print(f'the number of training images is {len(dataset_train)}')
    # val data
    dataset_val = VOCDataset(val_index_file_path, image_size, grid_number, number_bbox, number_classes)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)
    print(f'the number of validating images is {len(dataset_val)}')



if __name__ == '__main__':
    train()
