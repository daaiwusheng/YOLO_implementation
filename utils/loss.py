import torch
import torch.nn as nn
import torch.nn.functional as F

class LossBase(nn.Module):
    def __init__(self, number_grid=7, number_box=2, number_class=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(LossBase, self).__init__()
        self.number_gird = number_grid
        self.number_box = number_box
        self.number_class = number_class
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, box_1, box_2):
        def box_area(box):
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area

        area_1 = box_area(box_1)
        area_2 = box_area(box_2)

        inter_area =