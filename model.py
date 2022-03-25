import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_conv import AttentionConv, AttentionStem
from cbam import SpatialAttention, ChannelAttention

class Net(nn.Module):
    def __init__(self, channels, is_bn):
        super(Net, self).__init__()
        self.net_sm = Net_sm(channels, is_bn)
        self.net_ts = Net_ts(channels, is_bn)

        c = channels["sm"][-1] + channels["ts"][-1]

        self.li4 = nn.Linear(in_features=c, out_features=c//2)
        self.li5 = nn.Linear(in_features=c//2, out_features=c//4)
        self.li6 = nn.Linear(in_features=c//4, out_features=2)

    def forward(self, data, tdata):
        _, x = self.net_sm(data)
        _, y = self.net_ts(tdata)
        z = torch.cat([x, y], dim=1)
        z = F.relu(z)

        z = F.relu(self.li4(z))
        z = F.relu(self.li5(z))
        z = self.li6(z)
        prob = F.softmax(z, dim=1)

        return   z

class Net_sm(nn.Module):
    def __init__(self, channels, is_bn):
        super(Net_sm, self).__init__()
        self.is_bn = is_bn
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=7, padding=3, stride=3)
        self.bn1 = nn.BatchNorm3d(num_features=16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm3d(num_features=32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm3d(num_features=64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.ca = ChannelAttention(64, dim=3)
        self.sa = SpatialAttention(dim=3)

        self.li1 = nn.Linear(in_features=channels["sm"][0], out_features=channels["sm"][1])
        self.li2 = nn.Linear(in_features=channels["sm"][1], out_features=channels["sm"][2])
        self.li3 = nn.Linear(in_features=channels["sm"][2], out_features=2)

    def forward(self, data):
        is_bn = self.is_bn
        x = self.conv1(data)
        if is_bn:
            x = self.bn1(x)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        if is_bn:
            x = self.bn2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        if is_bn:
            x = self.bn3(x)
        x = self.pool3(F.relu(x))

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = x.flatten(start_dim=1)
        x = F.relu(self.li1(x))
        x = F.relu(self.li2(x))
        out = self.li3(x)
        prob = F.softmax(out, dim=1)
        return   out, x

class Net_ts(nn.Module):
    def __init__(self, channels, is_bn=True):
        super(Net_ts, self).__init__()
        self.is_bn = is_bn
        #self.conv1_ts = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, stride=3)
        self.conv1_ts = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, stride=3, dilation=1)
        self.bn1_ts = nn.BatchNorm1d(num_features=16)
        self.pool1_ts = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        #self.conv2_ts = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.conv2_ts = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=2, dilation=1)
        self.bn2_ts = nn.BatchNorm1d(num_features=32)
        self.pool2_ts = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.conv3_ts = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn3_ts = nn.BatchNorm1d(num_features=64)
        self.pool3_ts = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.ca_ts_2 = ChannelAttention(32, dim=1)
        self.sa_ts_2 = SpatialAttention(dim=1)
        self.ca_ts_3 = ChannelAttention(64, dim=1)
        self.sa_ts_3 = SpatialAttention(dim=1)

        self.li1_ts = nn.Linear(in_features=channels["ts"][0], out_features=channels["ts"][0]//2)
        self.li2_ts = nn.Linear(in_features=channels["ts"][0]//2, out_features=channels["ts"][0]//4)
        self.li3_ts = nn.Linear(in_features=channels["ts"][0]//4, out_features=2)

        # Self Attention
        self.self_attn_3 = Self_Attn_1d(32)
        self.self_attn_4 = Self_Attn_1d(64)

    def forward(self, tdata):
        is_bn = self.is_bn
        #tdata = self.self_attn(tdata) #2

        y = self.conv1_ts(tdata)
        if is_bn:
            y = self.bn1_ts(y)
        y = self.pool1_ts(F.relu(y))

        #y = self.self_attn(y) #1

        y = self.conv2_ts(y)
        if is_bn:
            y = self.bn2_ts(y)
        y = self.pool2_ts(F.relu(y))

        #y = self.self_attn_3(y)  # 3

        y = self.conv3_ts(y)
        if is_bn:
            y = self.bn3_ts(y)
        y = self.pool3_ts(F.relu(y))

        y = self.self_attn_4(y)  # 3

        y = self.ca_ts_3(y) * y
        y = self.sa_ts_3(y) * y

        y = y.flatten(start_dim=1)
        y = F.relu(self.li1_ts(y))
        y = F.relu(self.li2_ts(y))
        out = self.li3_ts(y)
        prob = F.softmax(out, dim=1)
        return   out, y


class Resnet_sm_ts(nn.Module):
    def __init__(self, res):
        super(Resnet_sm_ts, self).__init__()
        self.net_sm = Resnet_sm(res)
        self.net_ts = Resnet_ts(res)
        #self.net_ts = LSTM_ts(input_size=200, hidden_size=256)
        #self.net_ts = Resnet_ts_multi(res)
        #self.li1_c = nn.Linear(in_features=1024, out_features=512)
        #self.li2_c = nn.Linear(in_features=64, out_features=32)
        self.li3_c = nn.Linear(in_features=64, out_features=2)

    def forward(self, data, tdata):
        _,x = self.net_sm(data)
        _,y = self.net_ts(tdata)
        #z = torch.cat([x, y[:,-1,:]], dim=1)
        z = torch.cat([x, y], dim=1)
        z = F.relu(z)

        #z = F.relu(self.li1_c(z))
        #z = F.relu(self.li2_c(z))
        z = self.li3_c(z)

        return  z

class Resnet_sm(nn.Module):
    def __init__(self, res):
        super(Resnet_sm, self).__init__()
        self.net_sm = res(dims=3, num_class=32)
        #self.li1_sm = nn.Linear(in_features=1000, out_features=512)
        #self.li2_sm = nn.Linear(in_features=32, out_features=32)
        self.li3_sm = nn.Linear(in_features=32, out_features=2)

    def forward(self, data):
        x = F.relu(self.net_sm(data))
        #x = F.relu(self.li1_sm(x))
        #x = F.relu(self.li2_sm(x))
        z = self.li3_sm(x)

        return  z, x

class Resnet_ts(nn.Module):
    def __init__(self, res):
        super(Resnet_ts, self).__init__()
        self.net_ts = res(dims=1, num_class=32)
        #self.li1_ts = nn.Linear(in_features=1000, out_features=512)
        #self.li2_ts = nn.Linear(in_features=32, out_features=32)
        self.li3_ts = nn.Linear(in_features=32, out_features=2)

    def forward(self, tdata):
        x = F.relu(self.net_ts(tdata))
        #x = F.relu(self.li1_ts(x))
        #x = F.relu(self.li2_ts(x))
        z = self.li3_ts(x)

        return  z, x


class Self_Attn_1d(nn.Module):
    """ Self attention Layer"""
    # https://github.com/heykeetae/Self-Attention-GAN

    def __init__(self, in_dim):
        super(Self_Attn_1d, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X L)
            returns :
                out : self attention value + input feature
                attention: B X L (N is L)
        """
        m_batchsize, C, L = x.size()
        proj_query = self.query_conv(x).permute(0, 2, 1) # B X L X C//8
        proj_key = self.key_conv(x) # B X C//8 X L
        energy = torch.bmm(proj_query, proj_key) # B X L X L
        attention = self.softmax(energy)
        proj_value = self.value_conv(x) # B X C X L

        out = torch.bmm(proj_value, attention) # B X C x L
        out = out.view(m_batchsize, C, L)

        out = self.gamma * out + x
        return out

class ANet_sm_ts(nn.Module):
    def __init__(self, channels):
        super(ANet_sm_ts, self).__init__()
        self.net_sm = ANet_sm(channels)
        self.net_ts = ANet_ts(channels)

        c = channels["sm"][-1] + channels["ts"][-1]

        self.li4 = nn.Linear(in_features=c, out_features=c//2)
        self.li5 = nn.Linear(in_features=c//2, out_features=c//4)
        self.li6 = nn.Linear(in_features=c//4, out_features=2)

    def forward(self, data, tdata, is_bn=False):
        _, _, x = self.net_sm(data, is_bn)
        _, _, y = self.net_ts(tdata, is_bn)
        z = torch.cat([x, y], dim=1)
        z = F.relu(z)

        z = F.relu(self.li4(z))
        z = F.relu(self.li5(z))
        z = self.li6(z)
        prob = F.softmax(z, dim=1)

        return   z

class ANet_sm(nn.Module):
    def __init__(self, channels):
        super(ANet_sm, self).__init__()
        #self.conv1 = nn.Conv3d(in_channels=1, out_channels=36, kernel_size=7, stride=3, padding=3)
        self.conv1 = AttentionStem(in_channels=1, out_channels=36, kernel_size=4, stride=1, padding=1, dim=3)
        self.bn1 = nn.BatchNorm3d(num_features=36)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=3)
        self.bn2 = nn.BatchNorm3d(num_features=36)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=3)
        self.bn3 = nn.BatchNorm3d(num_features=36)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=3)
        self.bn4 = nn.BatchNorm3d(num_features=36)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5 = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=3)
        self.bn5 = nn.BatchNorm3d(num_features=36)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.li1 = nn.Linear(in_features=channels["sm"][0], out_features=512)
        self.li2 = nn.Linear(in_features=512, out_features=256)
        self.li3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, data, is_bn=True):
        x = self.conv1(data)
        if is_bn:
            x = self.bn1(x)
        x = self.pool1(F.relu(x))

        skip = x
        x = self.conv2(x)
        if is_bn:
            x = self.bn2(x)
        x = self.pool2(F.relu(x))

        x = self.conv3(x)
        if is_bn:
            x = self.bn3(x)
        x = self.pool3(F.relu(x))

        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x += F.max_pool3d(skip, kernel_size=4, stride=4)

        skip = x
        x = self.conv4(x)
        if is_bn:
            x = self.bn4(x)
        x = self.pool4(F.relu(x))

        x = self.conv5(x)
        if is_bn:
            x = self.bn5(x)
        x = self.pool5(F.relu(x))

        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x += F.max_pool3d(skip, kernel_size=4, stride=4)

        x = x.flatten(start_dim=1)
        x = F.relu(self.li1(x))
        x = F.relu(self.li2(x))
        out = self.li3(x)
        prob = F.softmax(out, dim=1)
        return   out, x

class ANet_ts(nn.Module):
    def __init__(self, channels, is_bn=True):
        super(ANet_ts, self).__init__()
        self.is_bn = is_bn
        self.conv1_ts = AttentionStem(in_channels=1, out_channels=36, kernel_size=4, stride=1, padding=1, dim=1)
        self.bn1_ts = nn.BatchNorm1d(num_features=36)
        self.pool1_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_ts = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=1)
        self.bn2_ts = nn.BatchNorm1d(num_features=36)
        self.pool2_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_ts = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=1)
        self.bn3_ts = nn.BatchNorm1d(num_features=36)
        self.pool3_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4_ts = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=1)
        self.bn4_ts = nn.BatchNorm1d(num_features=36)
        self.pool4_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5_ts = AttentionConv(in_channels=36, out_channels=36, kernel_size=3, padding=1, stride=1, dim=1)
        self.bn5_ts = nn.BatchNorm1d(num_features=36)
        self.pool5_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.li1_ts = nn.Linear(in_features=channels["ts"][0], out_features=channels["ts"][0]//2)
        self.li2_ts = nn.Linear(in_features=channels["ts"][0]//2, out_features=channels["ts"][0]//4)
        self.li3_ts = nn.Linear(in_features=channels["ts"][0]//4, out_features=2)

        # Self Attention
        self.self_attn_3 = Self_Attn_1d(32)
        self.self_attn_4 = Self_Attn_1d(64)

    def forward(self, tdata):
        is_bn = self.is_bn
        y = self.conv1_ts(tdata)
        if is_bn:
            y = self.bn1_ts(y)
        y = self.pool1_ts(F.relu(y))

        skip = y
        y = self.conv2_ts(y)
        if is_bn:
            y = self.bn2_ts(y)
        y = self.pool2_ts(F.relu(y))

        y = self.conv3_ts(y)
        if is_bn:
            y = self.bn3_ts(y)
        y = self.pool3_ts(F.relu(y))
        y = self.ca_ts1(y) * y

        y += F.max_pool1d(skip, kernel_size=4, stride=4)

        skip = y
        y = self.conv4_ts(y)
        if is_bn:
            y = self.bn4_ts(y)
        y = self.pool4_ts(F.relu(y))

        y = self.conv5_ts(y)
        if is_bn:
            y = self.bn5_ts(y)
        y = self.pool5_ts(F.relu(y))
        y = self.ca_ts2(y) * y
        y += F.max_pool1d(skip, kernel_size=4, stride=4)

        y = y.flatten(start_dim=1)
        y = F.relu(self.li1_ts(y))
        y = F.relu(self.li2_ts(y))
        out = self.li3_ts(y)
        prob = F.softmax(out, dim=1)
        return   out, y

class Net_2(nn.Module):
    def __init__(self, channels):
        super(Net_2, self).__init__()
        self.net_sm = Net_sm_2(channels, is_bn=True)
        self.net_ts = Net_ts_2(channels, is_bn=True)

        c = 512

        self.li4 = nn.Linear(in_features=c, out_features=c//2)
        self.li5 = nn.Linear(in_features=c//2, out_features=2)

    def forward(self, data, tdata):
        _, x = self.net_sm(data)
        _, y = self.net_ts(tdata)
        z = torch.cat([x, y], dim=1)

        z = F.relu(self.li4(z))
        z = self.li5(z)
        prob = F.softmax(z, dim=1)

        return   z

class Net_sm_2(nn.Module):
    def __init__(self, channels, is_bn=True):
        super(Net_sm_2, self).__init__()
        self.is_bn=is_bn
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm3d(num_features=32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm3d(num_features=32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm3d(num_features=32)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn4 = nn.BatchNorm3d(num_features=32)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn5 = nn.BatchNorm3d(num_features=32)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.ca1 = ChannelAttention(32, dim=3)
        self.sa1 = SpatialAttention(dim=3)
        self.ca2 = ChannelAttention(32, dim=3)
        self.sa2 = SpatialAttention(dim=3)
        self.ca3 = ChannelAttention(32, dim=3)
        self.sa3 = SpatialAttention(dim=3)
        self.ca4 = ChannelAttention(32, dim=3)
        self.sa4 = SpatialAttention(dim=3)

        self.li1 = nn.Linear(in_features=channels["sm"][0], out_features=512)
        self.li2 = nn.Linear(in_features=512, out_features=256)
        self.li3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, data):
        is_bn = self.is_bn
        x = self.conv1(data)
        if is_bn:
            x = self.bn1(x)
        x = self.pool1(F.relu(x))

        skip = x
        x = self.conv2(x)
        if is_bn:
            x = self.bn2(x)
        x = self.pool2(F.relu(x))
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x += F.max_pool3d(skip, kernel_size=2, stride=2)

        skip = x
        x = self.conv3(x)
        if is_bn:
            x = self.bn3(x)
        x = self.pool3(F.relu(x))
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x += F.max_pool3d(skip, kernel_size=2, stride=2)

        skip = x
        x = self.conv4(x)
        if is_bn:
            x = self.bn4(x)
        x = self.pool4(F.relu(x))
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        x += F.max_pool3d(skip, kernel_size=2, stride=2)

        skip = x
        x = self.conv5(x)
        if is_bn:
            x = self.bn5(x)
        x = self.pool5(F.relu(x))
        x = self.ca4(x) * x
        x = self.sa4(x) * x
        x += F.max_pool3d(skip, kernel_size=2, stride=2)

        x = x.flatten(start_dim=1)
        x = F.relu(self.li1(x))
        x = F.relu(self.li2(x))
        out = self.li3(x)
        prob = F.softmax(out, dim=1)
        return   out, x

class Net_ts_2(nn.Module):
    def __init__(self, channels, is_bn=True):
        super(Net_ts_2, self).__init__()
        self.is_bn = is_bn
        self.conv1_ts = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn1_ts = nn.BatchNorm1d(num_features=32)
        self.pool1_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_ts = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn2_ts = nn.BatchNorm1d(num_features=32)
        self.pool2_ts = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3_ts = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn3_ts = nn.BatchNorm1d(num_features=32)
        self.pool3_ts = nn.MaxPool1d(kernel_size=2, stride=2)

        self.ca_ts1 = ChannelAttention(32, dim=1)

        self.li1_ts = nn.Linear(in_features=channels["ts"][0], out_features=512)
        self.li2_ts = nn.Linear(in_features=512, out_features=256)
        self.li3_ts = nn.Linear(in_features=256, out_features=2)


    def forward(self, tdata):
        is_bn = self.is_bn
        y = self.conv1_ts(tdata)
        if is_bn:
            y = self.bn1_ts(y)
        y = self.pool1_ts(F.relu(y))

        skip = y
        y = self.conv2_ts(y)
        if is_bn:
            y = self.bn2_ts(y)
        y = self.pool2_ts(F.relu(y))
        y += F.max_pool1d(skip, kernel_size=2, stride=2)

        skip = y
        y = self.conv3_ts(y)
        if is_bn:
            y = self.bn3_ts(y)
        y = self.pool3_ts(F.relu(y))
        y = self.ca_ts1(y) * y
        y += F.max_pool1d(skip, kernel_size=2, stride=2)

        y = y.flatten(start_dim=1)
        y = F.relu(self.li1_ts(y))
        y = F.relu(self.li2_ts(y))
        out = self.li3_ts(y)
        prob = F.softmax(out, dim=1)
        return   out, y

class Net_3(nn.Module):
    def __init__(self, channels):
        super(Net_3, self).__init__()
        self.net_sm = Net_sm_3(channels)
        self.net_ts = Net_ts_3(channels)

        c = 512

        self.li4 = nn.Linear(in_features=c, out_features=c//2)
        self.li5 = nn.Linear(in_features=c//2, out_features=2)

    def forward(self, data, tdata, is_bn=False):
        _, _, x = self.net_sm(data, is_bn)
        _, _, y = self.net_ts(tdata, is_bn)
        z = torch.cat([x, y], dim=1)

        z = F.relu(self.li4(z))
        z = self.li5(z)
        prob = F.softmax(z, dim=1)

        return   z

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

class Net_sm_3(nn.Module):
    def __init__(self, channels):
        super(Net_sm_3, self).__init__()
        self.conv1 = conv3x3(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1, dims=3)
        self.bn1 = nn.BatchNorm3d(num_features=32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.li1 = nn.Linear(in_features=channels["sm"][0], out_features=512)
        self.li2 = nn.Linear(in_features=512, out_features=256)
        self.li3 = nn.Linear(in_features=256, out_features=2)

    def forward(self, data, is_bn=True):
        x = self.conv1(data)
        x = self.bn1(x)
        x = self.pool1(F.relu(x))

        for i in range(4):
            skip = x
            x = self.stage["conv1_{}".format(i)](x)
            x = self.stage["bn1_{}".format(i)](x)
            x = F.relu(x, inplace=True)
            x = self.stage["conv2_{}".format(i)](x)
            x = self.stage["bn2_{}".format(i)](x)
            x = F.relu(x, inplace=True)
            x = self.stage["conv3_{}".format(i)](x)
            x = self.stage["bn3_{}".format(i)](x)
            x = self.ca[i](x) * x
            x = self.sa[i](x) * x
            x = x + self.down["bn{}".format(i)](self.down["conv{}".format(i)](skip))
            x = F.relu(x, inplace=True)

        x = x.flatten(start_dim=1)
        x = F.relu(self.li1(x))
        x = F.relu(self.li2(x))
        out = self.li3(x)
        prob = F.softmax(out, dim=1)
        return   out, x


class Net_ts_3(nn.Module):
    def __init__(self, channels):
        super(Net_ts_3, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1_ts = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False, dilation=1)
        self.bn1_ts = nn.BatchNorm1d(num_features=32)
        self.pool1_ts = nn.MaxPool1d(kernel_size=2, stride=2)

        self.stage = []
        self.down = []
        self.ca = []
        for i in range(4):
            self.stage.append(nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False, dilation=1),
                nn.BatchNorm1d(num_features=16),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=2, bias=False, dilation=1),
                nn.BatchNorm1d(num_features=16),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, padding=0, stride=1, bias=False, dilation=1),
                nn.BatchNorm1d(num_features=32)
            ).to(device))
            self.down.append(nn.Sequential(
                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(num_features=32)
            ).to(device))
            self.ca.append(ChannelAttention(32, dim=1).to(device))

        self.li1_ts = nn.Linear(in_features=channels["ts"][0], out_features=512)
        self.li2_ts = nn.Linear(in_features=512, out_features=256)
        self.li3_ts = nn.Linear(in_features=256, out_features=2)


    def forward(self, tdata, is_bn):
        y = self.conv1_ts(tdata)
        y = self.bn1_ts(y)
        y = self.pool1_ts(F.relu(y))
        
        for i in range(4):
            skip = y
            y = self.stage[i](y)
            y = self.ca[i](y) * y
            y = y + self.down[i](skip)
            y = F.relu(y, inplace=True)

        y = y.flatten(start_dim=1)
        y = F.relu(self.li1_ts(y))
        y = F.relu(self.li2_ts(y))
        out = self.li3_ts(y)
        prob = F.softmax(out, dim=1)
        return   out, y


class Resnet_ts_multi(nn.Module):
    def __init__(self, res):
        super(Resnet_ts_multi, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.ca1 = ChannelAttention(4, dim=1)
        self.resnet = res(dims=1, input_channel=4)

        self.li1_ts = nn.Linear(in_features=1000, out_features=512)
        self.li2_ts = nn.Linear(in_features=512, out_features=256)
        self.li3_ts = nn.Linear(in_features=256, out_features=2)

    def forward(self, tdata):
        x = []
        x.append(tdata)
        x.append(F.avg_pool1d(tdata, kernel_size=3 ,padding=1, stride=1))
        x.append(F.avg_pool1d(tdata, kernel_size=5, padding=2, stride=1))
        x.append(F.avg_pool1d(tdata, kernel_size=7, padding=3, stride=1))
        x = torch.cat(x, dim=1)

        #x = F.relu(self.conv1(x))
        x = self.ca1(x)
        x = F.relu(self.resnet(x))
        x = F.relu(self.li1_ts(x))
        x = F.relu(self.li2_ts(x))
        out = self.li3_ts(x)
        prob = F.softmax(out, dim=1)
        return  out, x

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

class LSTM_ts(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(LSTM_ts, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.li = nn.Linear(hidden_size, 2)

    def forward(self, tdata):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(tdata)
        out = self.li(x[:,-1,:])
        return  out, x

class LSTM_parallel(nn.Module):
    def __init__(self, res, input_size=1, hidden_size=256):
        super(LSTM_parallel, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.net_ts = res(dims=1, num_class=32)
        self.dropout = nn.Dropout(p=0.5)
        #self.li1_sm = nn.Linear(in_features=1000, out_features=512)
        #self.li2_sm = nn.Linear(in_features=32, out_features=32)
        concat_channel = hidden_size + 32
        self.li1 = nn.Linear(in_features=concat_channel, out_features=concat_channel//2)
        self.li2 = nn.Linear(in_features=concat_channel//2, out_features=2)

    def forward(self, tdata):
        self.rnn.flatten_parameters()
        b, c, l= tdata.shape
        x = tdata.view(b, l, c)
        x, _ = self.rnn(x)
        x = self.dropout(x[:,-1,:])

        y = self.net_ts(tdata)

        z = torch.cat([x, y], dim=1)
        z = self.li1(z)
        z = self.li2(z)
        return z


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        #self.li1_sm = nn.Linear(in_features=1000, out_features=512)
        #self.li2_sm = nn.Linear(in_features=32, out_features=32)
        concat_channel = hidden_size
        self.li1 = nn.Linear(in_features=concat_channel, out_features=2)
        # self.li2 = nn.Linear(in_features=concat_channel//2, out_features=2)

    def forward(self, tdata):
        self.rnn.flatten_parameters()
        b, c, l= tdata.shape
        x = tdata.view(b, l, c)
        x, _ = self.rnn(x)
        x = self.dropout(x[:,-1,:])
        z = self.li1(x)
        # z = self.li2(z)
        return z

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

if __name__ == "__main__":
    from torchsummary import summary
    import cbam
    import resnet
    from fix_feature import Fix_ts_trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #c = {}
    #c["ts"] = [2048]
    #c["sm"] = [256]
    #net = Net_sm_2(channels=c).to(device)
    #net = Net_ts(channels={"ts":[256]},is_bn=False).to(device)
    # summary(net,(1,200), 30)
    # net = Resnet_sm(res=cbam.resnet_test3_234).to(device)
    # summary(net,(1,72,90,76),30)
    net = Resnet_ts(res=cbam.resnet_test3_234).to(device)
    summary(net,(1,1200),30)
    #print(net)
