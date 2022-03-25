import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import resnet

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

num = 0
#act = #Mish() #
act = nn.ReLU(inplace=True)


def save_feature(out, dim, name):
    name = str(num) + name
    filename = "test/tensor_{}_{}.mat".format(dim, name)
    mat = out.cpu().detach().numpy()
    scipy.io.savemat(filename, {"weight":mat})
    #torch.save(out, "test/tensor_{}_{}.pt".format(dim,name))
    print("save feature to {}".format(filename))

def conv_dims(dims):
    if dims == 3:
        f = nn.Conv3d
    elif dims == 2:
        f = nn.Conv2d
    elif dims == 1:
        f = nn.Conv1d
    else:
        raise Exception("dims is not in [1, 2, 3]")

    return f

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dims=3):
    """3x3 convolution with padding"""
    f = conv_dims(dims)
    return f(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, dims=3):
    """1x1 convolution"""
    f = conv_dims(dims)
    return f(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, dim=3):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        assert self.dim in [1, 2, 3], "dim [] not in [1, 2, 3]".format(self.dim)
        if dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

        conv = conv_dims(dim)
        hidden_planes = in_planes // 16 if in_planes // 16 > 1 else in_planes
        self.fc1   = conv(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = act
        self.fc2   = conv(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dim=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        conv = conv_dims(dim)
        self.conv1 = conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dims=3):
        super(BasicBlock, self).__init__()
        self.dims = dims
        if norm_layer is None:
            if dims == 3:
                norm_layer = nn.BatchNorm3d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            elif dims == 1:
                norm_layer = nn.BatchNorm1d
            else:
                raise ValueError("dims is not in [1, 2, 3]")
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dims=dims)
        self.bn1 = norm_layer(planes)
        self.relu = act
        self.conv2 = conv3x3(planes, planes, dims=dims)
        self.bn2 = norm_layer(planes)

        self.ca = ChannelAttention(planes,dim=dims)
        self.sa = SpatialAttention(dim=dims)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        global num
        num += 1

        #save_feature(out, self.dims,"_before")
        out_ca = self.ca(out)
        out_sa = self.sa(out)
        out = out_ca * out
        out = out_sa * out
        #save_feature(out_sa, self.dims, "_spatial")
        #save_feature(out_ca, self.dims, "_channel")
        #save_feature(out, self.dims, "_after")

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        #raise ValueError()
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dims=3):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.dims = dims
        if norm_layer is None:
            if dims == 3:
                norm_layer = nn.BatchNorm3d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            elif dims == 1:
                norm_layer = nn.BatchNorm1d
            else:
                raise Exception("dims is not in [1, 2, 3]")
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, dims=dims)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, dims=dims)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, dims=dims)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = act

        self.ca = ChannelAttention(planes * self.expansion,dim=dims)
        self.sa = SpatialAttention(dim=dims)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #save_feature(out, self.dims, str(self.inplanes)+"_before")
        out_ca = self.ca(out)
        out = out_ca * out

        if self.dims == 3:
            out_sa = self.sa(out)
            out = out_sa * out
            #save_feature(out_sa, self.dims, str(self.inplanes) + "_spatial")

        #save_feature(out_ca, self.dims, str(self.inplanes) + "_channel")
        #save_feature(out, self.dims, str(self.inplanes) + "_after")

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dims=3):
        super(ResNet, self).__init__()
        self.dims = dims
        if dims == 3:
            if norm_layer is None:
                norm_layer = nn.BatchNorm3d
            pool_layer = nn.MaxPool3d
            ada_pool_layer = nn.AdaptiveAvgPool3d
        elif dims == 1:
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            pool_layer = nn.MaxPool1d
            ada_pool_layer = nn.AdaptiveAvgPool1d
        else:
            raise Exception("dims is not in [1, 3]")
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        f = conv_dims(dims)
        self.conv1 = f(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = act
        self.maxpool = pool_layer(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = ada_pool_layer(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=self.dims),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dims=self.dims))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dims=self.dims))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_2(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dims=3, input_channel=1):
        super(ResNet_2, self).__init__()
        self.dims = dims
        if dims == 3:
            if norm_layer is None:
                norm_layer = nn.BatchNorm3d
            pool_layer = nn.MaxPool3d
            ada_pool_layer = nn.AdaptiveAvgPool3d
        elif dims == 2:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            pool_layer = nn.MaxPool2d
            ada_pool_layer = nn.AdaptiveAvgPool2d
        elif dims == 1:
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            pool_layer = nn.MaxPool1d
            ada_pool_layer = nn.AdaptiveAvgPool1d
        else:
            raise Exception("dims is not in [1, 3]")
        self._norm_layer = norm_layer
        self.input_channel = input_channel
        self.inplanes = 32#64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        f = conv_dims(dims)
        #self.conv1 = f(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = f(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = act
        self.maxpool = pool_layer(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = ada_pool_layer(1)
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=self.dims),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dims=self.dims))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dims=self.dims))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_3_234(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dims=3, input_channel=1):
        super(ResNet_3_234, self).__init__()
        self.dims = dims
        if dims == 3:
            if norm_layer is None:
                norm_layer = nn.BatchNorm3d
            pool_layer = nn.MaxPool3d
            ada_pool_layer = nn.AdaptiveAvgPool3d
        elif dims == 2:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            pool_layer = nn.MaxPool2d
            ada_pool_layer = nn.AdaptiveAvgPool2d
        elif dims == 1:
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            pool_layer = nn.MaxPool1d
            ada_pool_layer = nn.AdaptiveAvgPool1d
        else:
            raise Exception("dims is not in [1, 3]")
        self._norm_layer = norm_layer
        self.input_channel = input_channel
        self.inplanes = 32#64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        f = conv_dims(dims)
        #self.conv1 = f(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = f(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = act
        self.maxpool = pool_layer(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(resnet.BasicBlock, 32, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = ada_pool_layer(1)
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=self.dims),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dims=self.dims))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dims=self.dims))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_3_4(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dims=3, input_channel=1):
        super(ResNet_3_4, self).__init__()
        self.dims = dims
        if dims == 3:
            if norm_layer is None:
                norm_layer = nn.BatchNorm3d
            pool_layer = nn.MaxPool3d
            ada_pool_layer = nn.AdaptiveAvgPool3d
        elif dims == 2:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            pool_layer = nn.MaxPool2d
            ada_pool_layer = nn.AdaptiveAvgPool2d
        elif dims == 1:
            if norm_layer is None:
                norm_layer = nn.BatchNorm1d
            pool_layer = nn.MaxPool1d
            ada_pool_layer = nn.AdaptiveAvgPool1d
        else:
            raise Exception("dims is not in [1, 3]")
        self._norm_layer = norm_layer
        self.input_channel = input_channel
        self.inplanes = 32#64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        f = conv_dims(dims)
        #self.conv1 = f(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = f(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = act
        self.maxpool = pool_layer(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(resnet.BasicBlock, 32, layers[0])
        self.layer2 = self._make_layer(resnet.BasicBlock, 32, layers[1], stride=2,dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(resnet.BasicBlock, 32, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = ada_pool_layer(1)
        self.fc = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=self.dims),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dims=self.dims))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dims=self.dims))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, num_class=1000, dims=3,**kwargs):
    model = ResNet(block, layers, num_class, dims=dims, **kwargs)
    return model

def _resnet2(block, layers, num_class=1000, dims=3, input_channel=1, **kwargs):
    model = ResNet_2(block, layers, num_class, dims=dims, input_channel=input_channel,**kwargs)
    return model

def _resnet3_234(block, layers, num_class=1000, dims=3, input_channel=1, **kwargs):
    model = ResNet_3_234(block, layers, num_class, dims=dims, input_channel=input_channel, **kwargs)
    return model

def _resnet3_4(block, layers, num_class=1000, dims=3, input_channel=1, **kwargs):
    model = ResNet_3_4(block, layers, num_class, dims=dims, input_channel=input_channel, **kwargs)
    return model

def resnet_test3_234(pretrained=False, progress=True, dims=3, num_class=1000, input_channel=1, **kwargs):
    return _resnet3_234(BasicBlock, [1, 1, 1, 1], num_class=num_class, dims=dims, input_channel=input_channel, **kwargs)

def resnet_test3_4(pretrained=False, progress=True, dims=3, num_class=1000, input_channel=1, **kwargs):
    return _resnet3_4(BasicBlock, [1, 1, 1, 1], num_class=num_class, dims=dims, input_channel=input_channel, **kwargs)

def resnet_test(pretrained=False, progress=True, dims=3, num_class=1000, input_channel=1, **kwargs):
    return _resnet('', BasicBlock, [1, 1, 1, 1], pretrained, progress, num_class=num_class, dims=dims, **kwargs)
    #return _resnet('', Bottleneck, [1, 1, 1, 1], pretrained, progress, num_class=num_class, dims=dims, **kwargs)

def resnet_test2(pretrained=False, progress=True, dims=3, num_class=1000, input_channel=1, **kwargs):
    return _resnet2(BasicBlock, [1, 1, 1, 1], num_class=num_class, dims=dims, input_channel=input_channel, **kwargs)
    #return _resnet2(Bottleneck, [1, 1, 1, 1], num_class=num_class, dims=dims, input_channel=input_channel, **kwargs)

def resnet18(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, num_class=num_class, dims=dims, **kwargs)


def resnet34(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, num_class=num_class, dims=dims,
                   **kwargs)


def resnet50(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_class=num_class, dims=dims,
                   **kwargs)


def resnet101(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, num_class=num_class, dims=dims,
                   **kwargs)


def resnet152(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, dims=dims,  num_class=num_class,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, dims=dims,  num_class=num_class, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, dims=dims,  num_class=num_class,**kwargs)


def wide_resnet50_2(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, dims=dims,  num_class=num_class,**kwargs)


def wide_resnet101_2(pretrained=False, progress=True, dims=3, num_class=1000, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, dims=dims,  num_class=num_class, **kwargs)

