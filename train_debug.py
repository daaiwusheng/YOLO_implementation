import torch
from models.yolo_v1 import YoloVOne
from torchinfo import summary
from utils.dataset import VOCDataset
from torch.utils.data import DataLoader
import tqdm

print(f'CUDA Device Count: {torch.cuda.device_count()}')
device = torch.device('cuda' if torch.cuda.device_count() else 'cpu')
print(device)


def train():
    yolo_v_1 = YoloVOne(number_grid=7, number_bbox=2, number_classes=20)
    yolo_v_1 = yolo_v_1.to(device)
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

    number_epochs = 3
    for epoch in range(number_epochs):
        print(f'current epoch is {epoch}/{number_epochs}')
        # training process
        yolo_v_1.train()
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, (images, targets) in progress_bar:
            # load data per batch
            current_batch_size = images[0]
            images, targets = images.to(device), targets.to(device)
            predicted_results = yolo_v_1(images)
            # calculate loss
            # print(images.shape)
            # print(targets.shape)
            # print(predicted_results.shape)


if __name__ == '__main__':
    train()
