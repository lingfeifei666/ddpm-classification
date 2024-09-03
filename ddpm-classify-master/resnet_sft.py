#以原始图像特征为主特征
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
class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, photo_nc: int, stride: int = 1, downsample = None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = SPADEGroupNorm(planes, photo_nc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = SPADEGroupNorm(planes, photo_nc)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, photo_map):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, photo_map)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, photo_map)

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
    def __init__(self, num_classes: int, zero_init_residual: bool = False) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SPADEGroupNorm(self.inplanes, 128)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.convd1 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.convd2 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.convd3 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.convd4 = nn.Conv2d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        # self.fusion = TransFusion(upscale=2, img_size=(128, 128),
        #                window_size=8, img_range=1., depths=[6, 6, 6, 6],
        #                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

        self.layer1 = Basiclayer(64,  64,  128, stride=1)
        self.layer2 = Basiclayer(64,  128, 128, stride=2)
        self.layer3 = Basiclayer(128, 256, 128, stride=2)
        self.layer4 = Basiclayer(256, 512, 128, stride=2)


        # self.layer1 = self._make_layer(64,  layers[0], stride=1)#卷积输出的通道数
        # self.layer2 = self._make_layer(128, layers[1], stride=2)
        # self.layer3 = self._make_layer(256, layers[2], stride=2)
        # self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.do = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, x1):
        #先做7x7的卷积
        x  = self.conv1(x)
        x2 = self.convd1(x1)
        x  = self.bn1(x, x2)
        x  = self.relu(x)

        x= self.layer1(x, x2)

        x3 = self.convd2(x1)
        x = self.layer2(x, x3)

        x4 = self.convd3(x1)
        x = self.layer3(x, x4)

        x5 = self.convd4(x1)
        x = self.layer4(x, x5)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes, pretrained: bool = False, path = None, progress: bool = True) -> ResNet:
    '''
    pretrained决定是否用预训练的参数
    如果没有指定自己预训练模型的路径，就从官方网站下载resnet18-f37072fd.pth for name, layer in model.named_children():
        print(f"Layer name: {name}\nLayer structure: {layer}\n")
    progress显示下载进度
    '''
    model=ResNet(num_classes, zero_init_residual= True)
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
    x1 = torch.randn(8, 128, 256, 256)
    model = resnet18(2)
    for name, layer in model.named_children():
        print(f"Layer name: {name}\nLayer structure: {layer}\n")
    # model = resnet18(2)
    # a = model.state_dict()
    o = model(x, x1)
    print(o.shape)