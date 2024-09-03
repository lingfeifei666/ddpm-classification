import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from TransformerFusion import TransFusion
import torch.nn.functional as F

class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps=1e-5):#norm_nc是结果输出的通道数，label_nc是分割图的输入通道数
        super().__init__()

        self.norm = nn.GroupNorm(norm_nc, norm_nc, affine=False)# 32/16

        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        return x * (1 + gamma) + beta

##resnet每个残差链接模块
class BasicBlock(nn.Module):#将原图特征和扩散模型结合作为归一化层
    def __init__(self, inplanes: int, planes: int, photo_nc: int, stride: int = 1, downsample = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(planes)
        self.bn1 = SPADEGroupNorm(planes, photo_nc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(planes)
        self.bn2 = SPADEGroupNorm(planes, photo_nc)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, photo_map):
        identity = x

        out = self.conv1(x)
        out = self.bn11(out)
        out = self.bn1(out, photo_map)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn12(out)
        out = self.bn2(out, photo_map)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)
        return out

class BasicBlock2(nn.Module):#常规归一化层batchnorm
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None) -> None:
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Basiclayer(nn.Module):
    def __init__(self, inplanes, planes, photo_nc, stride, downsample = None):
        super(Basiclayer, self).__init__()

        # 创建多个 BasicBlock 层
        if stride != 1:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        self.block1 = BasicBlock(inplanes, planes, photo_nc, stride, downsample)
        self.block2 = BasicBlock(planes, planes, photo_nc)

        # 可以继续添加其他层或模块...

    def forward(self, x, photo_map):
        # 模型的前向传播过程
        out = self.block1(x, photo_map)
        out = self.block2(out, photo_map)

        # 可以添加更多层的连接和处理...
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes: int, layers: list, zero_init_residual: bool = False) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SPADEGroupNorm(self.inplanes, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = Basiclayer(64,  64,  64, stride=1)
        self.layer2 = Basiclayer(64,  128, 64, stride=2)
        self.layer3 = Basiclayer(128, 256, 128, stride=2)
        self.layer4 = Basiclayer(256, 512, 256, stride=2)


        self.convd1 = nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.layerk1 = self._make_layer(64,  layers[0], stride=1)#卷积输出的通道数
        self.layerk2 = self._make_layer(128, layers[1], stride=2)
        self.layerk3 = self._make_layer(256, layers[2], stride=2)
        self.layerk4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.do = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.do = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock2):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = []
        layers.append(BasicBlock2(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock2(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, x1):
        #先做7x7的卷积
        x = self.conv1(x)
        x2 = self.convd1(x1)
        print(x.shape, x2.shape)
        x = self.bn1(x, x2)
        x = self.relu(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x = self.layer1(x, x2)
        x3 = self.layerk1(x2)
        # print(x.shape, x3.shape)
        x = self.layer2(x, x3)
        x4 = self.layerk2(x3)

        x = self.layer3(x, x4)
        x5 = self.layerk3(x4)

        x = self.layer4(x, x5)
        x6 = self.layerk4(x5)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x6 = self.avgpool1(x6)
        x6 = torch.flatten(x6, 1)
        x6 = self.fc1(x6)
        return x, x6


def resnet18(num_classes, pretrained: bool = False, path = None, progress: bool = True) -> ResNet:
    '''
    pretrained决定是否用预训练的参数
    如果没有指定自己预训练模型的路径，就从官方网站下载resnet18-f37072fd.pth for name, layer in model.named_children():
        print(f"Layer name: {name}\nLayer structure: {layer}\n")
    progress显示下载进度
    '''
    model=ResNet(num_classes, [2, 2, 2, 2], zero_init_residual=True)
    if pretrained:
        if not path:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',progress=progress)
        else:
            pretrained_state_dict = torch.load(path)
        state_dict=model.state_dict()
        for k in pretrained_state_dict:
            if k in state_dict and k not in ['fc.weight', 'fc.bias']:
                state_dict[k] = pretrained_state_dict[k].data
        model.load_state_dict(state_dict)
    return model

if __name__=='__main__':

    x = torch.randn(8, 3, 256, 256)
    x1 = torch.randn(8, 256, 128, 128)
    model = resnet18(2)
    for name, layer in model.named_children():
        print(f"Layer name: {name}\nLayer structure: {layer}\n")
    # model = resnet18(2)
    # a = model.state_dict()
    o, o1 = model(x, x1)
    print(o.shape, o1.shape)