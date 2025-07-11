"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv2_T_w', 'mobilenetV2', 'mobilenetV2_aux']

BN = None


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.names = ['0', '1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        t = x

        if self.use_res_connect:
            return t + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """mobilenetV2"""
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],

            [T, 32, 3, 2],

            [T, 64, 4, 2],
            [T, 96, 3, 1],

            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel,1)

        # building inverted residual blocks
        self.blocks = nn.ModuleList([])
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # building classifier
        #self.classifier = nn.Sequential(
        #    # nn.Dropout(0.5),
        #    nn.Linear(self.last_channel, feature_dim),
        #)
        self.classifier = nn.Linear(self.last_channel, feature_dim)

        self._initialize_weights()

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def forward(self, x, is_feat=False, preact=False):

        out = self.conv1(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        f1 = out
        out = self.blocks[2](out)
        f2 = out
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        f3 = out
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        f4 = out

        out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.classifier(out)

        if is_feat:
            return [f1, f2, f3, f4], out
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Auxiliary_Classifier(nn.Module):
    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(Auxiliary_Classifier, self).__init__()

        self.remove_avg = remove_avg
        self.width_mult = width_mult

        # 只保留3个分支
        interverted_residual_setting1 = [
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]
        self.block_extractor1 = self._make_layer(input_channel=12, interverted_residual_setting=interverted_residual_setting1)

        interverted_residual_setting2 = [
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]
        self.block_extractor2 = self._make_layer(input_channel=16, interverted_residual_setting=interverted_residual_setting2)

        interverted_residual_setting3 = [
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]
        self.block_extractor3 = self._make_layer(input_channel=48, interverted_residual_setting=interverted_residual_setting3)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.conv2_1 = conv_1x1_bn(160, self.last_channel)
        self.conv2_2 = conv_1x1_bn(160, self.last_channel)
        self.conv2_3 = conv_1x1_bn(160, self.last_channel)

        self.fc1 = nn.Linear(self.last_channel, feature_dim)
        self.fc2 = nn.Linear(self.last_channel, feature_dim)
        self.fc3 = nn.Linear(self.last_channel, feature_dim)

        self._initialize_weights()

    def _make_layer(self, input_channel, interverted_residual_setting):
        # building inverted residual blocks
        blocks = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * self.width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            blocks.append(nn.Sequential(*layers))

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        ss_logits = []
        ss_feats = []
        # 只处理前3个特征
        for i in range(3):
            idx = i + 1
            out = getattr(self, 'block_extractor'+str(idx))(x[i])
            out = getattr(self, 'conv2_'+str(idx))(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            ss_feats.append(out)
            out = getattr(self, 'fc'+str(idx))(out)
            ss_logits.append(out)
        return ss_feats, ss_logits


class MobileNetv2_Auxiliary(nn.Module):
    def __init__(self, T, W, feature_dim=100):
        super(MobileNetv2_Auxiliary, self).__init__()
        self.backbone = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
        self.auxiliary_classifier = Auxiliary_Classifier(T=T, feature_dim= feature_dim, width_mult=W)

    def forward(self, x, grad=False, att=False):
        feats, logit = self.backbone(x, is_feat=True)
        # 只取前3个特征
        feats = feats[:3]
        if grad is False:
            for i in range(len(feats)):
                feats[i] = feats[i].detach()
        ss_feats, ss_logits = self.auxiliary_classifier(feats)
        if att is False:
            return logit, ss_logits
        else:
            return logit, ss_logits, feats


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def mobilenetV2(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)
def mobilenetV2_aux(num_classes):
    return MobileNetv2_Auxiliary(6, 0.5, num_classes)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = mobilenetV2_aux(100)
    logit, ss_logits = net(x)
    print(logit)
    print(ss_logits)
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))
