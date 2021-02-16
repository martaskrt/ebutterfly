
import torch
import torchvision.models as models
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math


class Satellite_Model(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(Satellite_Model, self).__init__()
        self.inc_bias = False
        self.num_classes = num_classes
  
        self.lin = nn.Sequential(nn.Linear(in_channels, int(in_channels/4)),
                                            nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        self.class_emb = nn.Linear(int(in_channels/4), num_classes, bias=self.inc_bias)


    def forward(self, x):
        feats = self.lin(x)
        #emb = self.class_emb(feats)
        return feats
