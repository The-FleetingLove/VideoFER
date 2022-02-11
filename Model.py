import torch.nn as nn
import torchvision.models as models


# ResNet18 Model
class Res18Feature(nn.Module):  # 网络架构定义
    def __init__(self, pretrained=True, drop_rate=0):  # num_classes参数设置：FER2013Plus数据集下训练设置为8， RAF-DB数据集设置为7
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate

        resnet = models.resnet18(pretrained)  # 调用预训练模型
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1
        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, 2)  # new fc layer 512x2

    def forward(self, x):
        x = self.features(x)
        if self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
