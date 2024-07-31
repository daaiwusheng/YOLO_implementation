from torch.utils.data import Dataset
import cv2
import copy

class VOCDataset(Dataset):
    def __init__(self, index_file_path, image_size=448, grid_size=7, number_bbox=2, number_classes=20):
        super(VOCDataset, self).__init__()
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
                    self.boxes_list.append([boxes])
                    self.labels_list.append(labels)
                f_label_file.close()
                count += 1
                print(line)
        f_data_index.close()
        print(f'{count}, images number is {len(self.image_paths_list)}, {len(self.boxes_list)},  {len(self.labels_list)}')

    def __len__(self):
        return len(self.image_paths_list)

    def __getitem__(self, index):
        current_image_path = self.image_paths_list[index]
        current_image = cv2.imread(current_image_path)
        current_boxes = copy.deepcopy(self.boxes_list[index])



if __name__ == '__main__':
    train_dataset = VOCDataset('/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex/train.txt')

