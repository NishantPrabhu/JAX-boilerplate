
import jax
import haiku as hk
import jax.nn as nn
import jax.numpy as jnp
from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import module
from haiku._src import pool

# Convention redefinition
hk.Module = module.Module
hk.Conv2d = conv.Conv2d
hk.Linear = basic.Linear
hk.max_pool = pool.max_pool
hk.avg_pool = pool.avg_pool
hk.Sequential = basic.Sequential
hk.BatchNorm = batch_norm.BatchNorm
del basic, batch_norm, module, pool, conv


def conv3x3(out_planes, stride=1, groups=1, dilation=(1, 1)):
    return hk.Conv2d(out_planes, kernel_shape=3, stride=stride, padding=dilation, dilation=dilation,
                     feature_group_count=groups, with_bias=False)

def conv1x1(out_planes, stride=1):
    return hk.Conv2d(out_planes, kernel_shape=1, stride=stride, with_bias=False)


class BasicBlock(hk.Module):
    expansion: int = 1
    bn_config: dict = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}

    def __init__(self, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=(1, 1), bn_config=None, name=None):
        super(BasicBlock, self).__init__(name=name)
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation[0] > 1 or dilation[1] > 1:
            raise NotImplementedError("BasicBlock only supports dilation < (1, 1)")
        if bn_config is not None and isinstance(bn_config, dict):
            self.bn_config.update(bn_config)

        self.conv1 = conv3x3(planes, stride)
        self.bn1 = hk.BatchNorm(**self.bn_config)
        self.conv2 = conv3x3(planes)
        self.bn2 = hk.BatchNorm(**self.bn_config)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = nn.relu(out)
        return out


class Bottleneck(hk.Module):
    expansion = 4
    bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}

    def __init__(self, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=(1, 1), bn_config=None, name=None):
        super(Bottleneck, self).__init__(name=name)
        width = int(planes * base_width / 64.0)

        self.conv1 = conv1x1(width)
        self.bn1 = hk.BatchNorm(**self.bn_config)
        self.conv2 = conv3x3(width, stride, groups, dilation)
        self.bn2 = hk.BatchNorm(**self.bn_config)
        self.conv3 = conv1x1(planes * self.expansion)
        self.bn3 = hk.BatchNorm(**self.bn_config)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        identity = x
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = nn.relu(out)
        return out


class ResNet(hk.Module):
    bn_config = {"create_scale": True, "create_offset": True, "decay_rate": 0.999}

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=[False, False, False], reduce_first_conv=False, bn_config=None, name=None):
        super(ResNet, self).__init__(name=name)
        self.inplanes = 64
        self.dilation = (1, 1)
        self.groups = groups
        self.base_width = width_per_group

        if not reduce_first_conv:
            self.conv1 = hk.Conv2d(self.inplanes, kernel_shape=7, stride=2, padding=(3, 3), with_bias=False)
        else:
            self.conv1 = hk.Conv2d(self.inplanes, kernel_shape=3, stride=1, padding=(1, 1), with_bias=False)
        self.bn1 = hk.BatchNorm()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, stride=2, dilate=replace_stride_with_dilation[2])
        self.fc = hk.Linear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation = (self.dilation[0]*stride, self.dilation[1]*stride)
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = hk.Sequential([
                conv1x1(planes * block.expansion, stride),
                hk.BatchNorm()
            ])
        layers = []
        layers.append(block(planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return hk.Sequential(layers)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = hk.max_pool(x, window_shape=(3, 3), strides=2, padding="SAME")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = hk.avg_pool(x, window_shape=(x.shape[1], x.shape[2]), strides=1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)

def _resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
