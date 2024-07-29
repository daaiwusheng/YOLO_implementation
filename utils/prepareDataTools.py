import os
import math


def data_index(path_trainval_file, path_data_index_files, image_and_label_dir, train=9, val=1):
    mod = math.floor((train + val) / val)
    index_file_name = ''
    train_data_index_path = path_data_index_files + '/train.txt'
    val_data_index_path = path_data_index_files + '/val.txt'
    os.makedirs(path_data_index_files, exist_ok=True)

    train_image_number = 0
    val_image_number = 0
    count = 1
    with open(path_trainval_file, 'r') as file_in, open(train_data_index_path, 'w') as f_train_out, open(
            val_data_index_path, 'w') as f_val_out:
        for line in file_in:
            image_file_name = line.strip()
            full_path = os.path.join(image_and_label_dir, image_file_name)
            if count % mod != 0:  # put in train
                f_train_out.write(full_path + '\n')
                train_image_number += 1
            else:  # put in val
                f_val_out.write(full_path + '\n')
                val_image_number += 1
            count += 1
    f_val_out.close()
    f_train_out.close()
    print(f"{index_file_name} total image number is {count}")
    print(f"train images total image number is {train_image_number}")
    print(f"val images total image number is {val_image_number}")


if __name__ == '__main__':
    data_index('/home/wangyu/dataset/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
               '/home/wangyu/dataset/VOCdevkit/VOC2012/YoloLabelsIndex',
               '/home/wangyu/dataset/VOCdevkit/VOC2012/YoloImagsAndLabels')
