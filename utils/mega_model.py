import torch
import torchvision.models as models
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class MegaModel(nn.Module):
    def __init__(self, num_classes, num_filts, img_model, geo_model):
        super(MegaModel, self).__init__()
        self.inc_bias = False
        self.num_classes = num_classes
        
        self.cnn = nn.Sequential(img_model.backbone, img_model.avgpool)
        self.img_classifier = img_model.classifier
        self.feats = nn.Sequential(geo_model.feats, geo_model.class_emb)
        self.combine_loc_img = nn.Sequential(nn.Linear(num_classes*3, 256),
                                                nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        self.class_emb = nn.Linear(256, num_classes, bias=self.inc_bias)
    def forward(self, imgs, x, return_feats=False):
        loc_emb = x
        img_emb = imgs
        multiplied_emb = loc_emb * img_emb
        multiplied_emb /= multiplied_emb.sum(1, keepdim=True)
        combined = torch.cat((loc_emb, img_emb, multiplied_emb), 1)
        class_pred = self.class_emb(self.combine_loc_img(combined))
        return class_pred



