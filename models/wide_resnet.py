import torchvision.models as models
from torch import nn


class WideResnet(nn.Module):
    def __init__(self, config, loss_type):
        super(WideResnet, self).__init__()
        self.num_classes = config.num_classes
        try:
            self.dropout = nn.Dropout(p=config.dropout)
        except AttributeError:
            print('no dropout')
            self.dropout = nn.Dropout(p=0)
        self.loss_type = loss_type
        try:
            self.depth = config.depth
        except AttributeError:
            print('Default depth set to 50.')
            self.depth = 50
        if self.depth == 101:
            self.backbone = models.wide_resnet101_2(config.pretrained)
        else:
            self.backbone = models.wide_resnet50_2(config.pretrained)
        out_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        embedding = self.avgpool(x)
        x = embedding.view(embedding.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        if self.loss_type == "geometric":
            return(x, embedding)
        else:
            return(x)
