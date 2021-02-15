import torchvision.models as models
from torch import nn


class Resnet(nn.Module):
    def __init__(self, num_classes):
        super(Resnet, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=0)
        self.backbone = models.resnet50(True)
        out_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        embedding = self.avgpool(x)
        x = embedding.view(embedding.size(0), -1)
        x = self.classifier(x)
        x = self.dropout(x)
        return(x)
