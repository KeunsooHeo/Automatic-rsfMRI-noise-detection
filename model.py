import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import SpatialAttention, ChannelAttention

class Resnet_sm_ts(nn.Module):
    def __init__(self, res):
        super(Resnet_sm_ts, self).__init__()
        self.net_sm = Resnet_sm(res)
        self.net_ts = Resnet_ts(res)
        self.li3_c = nn.Linear(in_features=64, out_features=2)

    def forward(self, data, tdata):
        _,x = self.net_sm(data)
        _,y = self.net_ts(tdata)
        z = torch.cat([x, y], dim=1)
        z = F.relu(z)

        z = self.li3_c(z)

        return  z

class Resnet_sm(nn.Module):
    def __init__(self, res):
        super(Resnet_sm, self).__init__()
        self.net_sm = res(dims=3, num_class=32)
        self.li3_sm = nn.Linear(in_features=32, out_features=2)

    def forward(self, data):
        x = F.relu(self.net_sm(data))
        z = self.li3_sm(x)

        return  z, x

class Resnet_ts(nn.Module):
    def __init__(self, res):
        super(Resnet_ts, self).__init__()
        self.net_ts = res(dims=1, num_class=32)
        self.li3_ts = nn.Linear(in_features=32, out_features=2)

    def forward(self, tdata):
        x = F.relu(self.net_ts(tdata))
        z = self.li3_ts(x)

        return  z, x


def conv_dims(dims):
    if dims == 3:
        f = nn.Conv3d
    elif dims == 1:
        f = nn.Conv1d
    else:
        raise Exception("dims is not in [1, 3]")

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


class Resnet_sm_gradcam(nn.Module):
    def __init__(self, res):
        super(Resnet_sm_gradcam, self).__init__()
        self.net_sm = res(dims=3, num_class=32)
        #self.li1_sm = nn.Linear(in_features=1000, out_features=512)
        #self.li2_sm = nn.Linear(in_features=32, out_features=32)
        self.li3_sm = nn.Linear(in_features=32, out_features=2)

    def forward(self, data):
        x = F.relu(self.net_sm(data))
        #x = F.relu(self.li1_sm(x))
        #x = F.relu(self.li2_sm(x))
        z = self.li3_sm(x)

        return  z

class Resnet_ts_gradcam(nn.Module):
    def __init__(self, res):
        super(Resnet_ts_gradcam, self).__init__()
        self.net_ts = res(dims=1, num_class=32)
        #self.li1_sm = nn.Linear(in_features=1000, out_features=512)
        #self.li2_sm = nn.Linear(in_features=32, out_features=32)
        self.li3_ts = nn.Linear(in_features=32, out_features=2)
        self.net_ts.layer4.register_forward_hook(self.forward_hook)
        self.net_ts.layer4.register_backward_hook(self.backward_hook)

    def forward(self, data):
        x = F.relu(self.net_ts(data))
        #x = F.relu(self.li1_sm(x))
        #x = F.relu(self.li2_sm(x))
        z = self.li3_ts(x)

        return  z

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])


class Resnet_fusion(nn.Module):
    def __init__(self, res):
        super(Resnet_fusion, self).__init__()
        self.net_sm1 = res(dims=3, num_class=32)
        self.net_ts1 = res(dims=1, num_class=32)
        self.net_sm2 = res(dims=3, num_class=32)
        self.net_ts2 = res(dims=1, num_class=32)
        self.li_smts = nn.Linear(64, 32)
        self.li1 = nn.Linear(96, 32)
        self.li2 = nn.Linear(32, 2)

    def forward(self, data, tdata):
        x1 = self.net_sm1(data)
        y1 = self.net_ts1(tdata)
        x2 = self.net_sm2(data)
        y2 = self.net_ts2(tdata)
        z2 = torch.cat([x2, y2], dim=1)
        z2 = self.li_smts(z2)
        z = torch.cat([x1, y1, z2], dim=1)
        z = self.li1(z)
        z = self.li2(z)

        return  z