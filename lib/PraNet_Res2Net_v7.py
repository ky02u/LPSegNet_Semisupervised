import torch
import torch.nn             as nn
import torch.nn.functional  as F

from .Res2Net_v1b   import res2net50_v1b_26w_4s
from torch.autograd import Variable

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv   = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn     = nn.BatchNorm2d(out_planes)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        # self.branch4 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        # )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # x4 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2   = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3   = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4          = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5          = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x    = self.conv4(x3_2)
        x    = self.conv5(x)
        return x


class aggregation_LP(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation_LP, self).__init__()
        channel = channel*4
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4          = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5          = nn.Conv2d(channel, 1, 1)

    def forward(self, x1):
        x1_ = self.conv_upsample1(self.upsample(x1))
        x    = self.conv4(x1_)
        x    = self.conv5(x)
        # x2    = self.conv5(self.upsample(x1))
        return x

class aggregation_LP_2(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation_LP_2, self).__init__()
        channel = channel
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4          = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5          = nn.Conv2d(channel, 1, 1)

    def forward(self, x1):
        x1_ = self.conv_upsample1(self.upsample(x1))
        x    = self.conv4(x1_)
        x    = self.conv5(x)
        return x

class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, use_attention='PCM', mode_cls = 'max_pooling'):
        super(PraNet, self).__init__()
        self.relu = nn.ReLU(True)
        self.use_attention = use_attention
        self.mode_cls = mode_cls
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module - Multihead ----
        self.rfb2_0 = RFB_modified(512, channel)
        self.rfb2_1 = RFB_modified(1024, channel)
        self.rfb2_2 = RFB_modified(1024, channel)
        self.rfb2_3 = RFB_modified(1024, channel)
        self.rfb2_4 = RFB_modified(1024, channel)
        # self.rfb2_5 = RFB_modified(1024, channel)

        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb3_2 = RFB_modified(1024, channel)
        self.rfb3_3 = RFB_modified(1024, channel)
        self.rfb3_4 = RFB_modified(1024, channel)
        # self.rfb3_5 = RFB_modified(1024, channel)

        self.rfb4_1 = RFB_modified(2048, channel)
        self.rfb4_2 = RFB_modified(2048, channel)
        self.rfb4_3 = RFB_modified(2048, channel)
        self.rfb4_4 = RFB_modified(2048, channel)
        # self.rfb4_5 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        self.agg_LP = aggregation_LP(channel)
        # self.agg_LP_2 = aggregation_LP_2(channel) #only for deep supervision

        # --- GAP ---- 
        self.conv_gap = BasicConv2d(2048, 2, kernel_size=1,dilation=1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # --- dense layer to classify --- 
        self.output = nn.Linear(1000,1)
        self.dense = nn.Linear(1024, 2)

        # --- extract CAM and PCM -------
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.fc8 = nn.Conv2d(2048, 2, 1, dilation=3, bias=False)
        self.f_conv2 = torch.nn.Conv2d(32, 32, 1, bias=False)
        self.f_conv3 = torch.nn.Conv2d(32, 32, 1, bias=False)
        self.f_conv4 = torch.nn.Conv2d(2048, 64, 1, bias=False)
        self.f9 = torch.nn.Conv2d(128, 192, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f_conv2.weight)
        torch.nn.init.kaiming_normal_(self.f_conv3.weight)
        torch.nn.init.kaiming_normal_(self.f_conv4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        # --- include gamma ---
        # self.gamma_1 = nn.Parameter(torch.zeros(1))
        # self.gamma_2 = nn.Parameter(torch.zeros(1))
        # self.gamma_3 = nn.Parameter(torch.zeros(1))
        # self.gamma_4 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88

        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # ---- RFB ------
        v = self.rfb2_0(x2)
        value_1 = self.rfb2_1(x3)
        value_2 = self.rfb2_2(x3)
        value_3 = self.rfb2_3(x3)
        value_4 = self.rfb2_4(x3)
        # value_5 = self.rfb2_5(x3)

        key_1 = self.rfb3_1(x3)
        key_2 = self.rfb3_2(x3)
        key_3 = self.rfb3_3(x3)
        key_4 = self.rfb3_4(x3)
        # key_5 = self.rfb3_5(x3)

        query_1 = self.rfb4_1(x4)       
        query_2 = self.rfb4_2(x4)       
        query_3 = self.rfb4_3(x4)
        query_4 = self.rfb4_4(x4)
        # query_5 = self.rfb4_5(x4)

        # ra5_feat_PCM = self.agg1(x4_rfb_new, x3_rfb_new, x2_rfb_new)
        # ra5_LP = self.agg_LP(value)
        ra5_feat = self.agg1(query_1, key_1, v)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        if self.use_attention == 'PCM':
            n,c,h,w = ra5_feat.size()
            
            x3_feat = F.interpolate(x4,(h,w),mode='bilinear',align_corners=True)
            x_s_1   = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)

            f_2_1  = F.interpolate(key_1,(h,w),mode='bilinear',align_corners=True)
            f_3_1  = F.interpolate(query_1,(h,w),mode='bilinear',align_corners=True)
            head_1 = torch.cat([x_s_1,f_2_1,f_3_1], dim=1)
            pcm_1  = self.PCM(value_1, head_1)

            f_2_2  = F.interpolate(key_2,(h,w),mode='bilinear',align_corners=True)
            f_3_2  = F.interpolate(query_2,(h,w),mode='bilinear',align_corners=True)
            head_2 = torch.cat([x_s_1,f_2_2,f_3_2], dim=1)
            pcm_2  = self.PCM(value_2, head_2)

            f_2_3  = F.interpolate(key_3,(h,w),mode='bilinear',align_corners=True)
            f_3_3  = F.interpolate(query_3,(h,w),mode='bilinear',align_corners=True)
            head_3 = torch.cat([x_s_1,f_2_3,f_3_3], dim=1)
            pcm_3  = self.PCM(value_3, head_3 )

            f_2_4  = F.interpolate(key_4,(h,w),mode='bilinear',align_corners=True)
            f_3_4  = F.interpolate(query_4,(h,w),mode='bilinear',align_corners=True)
            head_4 = torch.cat([x_s_1,f_2_4,f_3_4], dim=1)
            pcm_4  = self.PCM(value_4, head_4)

            # f_2_5 = F.interpolate(key_5,(h,w),mode='bilinear',align_corners=True)
            # f_3_5 = F.interpolate(query_5,(h,w),mode='bilinear',align_corners=True)
            # x_s_5 = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
            # head_5 = torch.cat([x_s_1,f_2_5,f_3_5], dim=1)
            # pcm_5 = self.PCM(value_5, head_5)
            
            pcm_t = torch.cat([pcm_1,pcm_2,pcm_3,pcm_4], dim=1)
            pcm = self.agg_LP(pcm_t)
            pcm_ = F.interpolate(pcm, scale_factor=4, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
            
            #dense supervision
            # pcm1 = self.agg_LP_2(F.interpolate(pcm_1, scale_factor=4, mode='bilinear')   )
            # pcm2 = self.agg_LP_2(F.interpolate(pcm_2, scale_factor=4, mode='bilinear')  )
            # pcm3 = self.agg_LP_2(F.interpolate(pcm_3, scale_factor=4, mode='bilinear')  )
            # pcm4 = self.agg_LP_2(F.interpolate(pcm_4, scale_factor=4, mode='bilinear')  )
            
        if self.mode_cls == 'max_pooling':
            B,C,W,H = pcm_.shape
            max_pool = nn.MaxPool2d((W, H))
            # segmentation = torch.round(torch.sigmoid(lateral_map_5))
            classifcation = max_pool(torch.sigmoid(pcm_))

        if self.mode_cls == 'adaptive_avg_pooling':
            conv_gap = self.conv_gap(x4)
            gap = self.gap(conv_gap)
            softmax = nn.Softmax(dim=1)
            classifcation = softmax(gap)

        if self.mode_cls == 'softmax':
            a = self.avg(pcm_)
            avg = a.view(a.size(0), -1)
            dense  = self.fc(avg)
            classifcation = dense.unsqueeze(2).unsqueeze(3)

        # print(classifcation.shape)
        # print(classifcation)
        
        return pcm_, classifcation

    def PCM(self, value, attention):
        
        n,c,h,w = attention.size()
        
        proj_value = F.interpolate(value, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        # attention = self.f9(attention)
        attention_ = attention
        attention = attention.view(n,-1,h*w)
        attention = attention/(torch.norm(attention,dim=1,keepdim=True)+1e-5)

        attention_matrix = torch.matmul(attention.transpose(1,2), attention)
        attention_matrix = attention_matrix/(torch.sum(attention_matrix,dim=1,keepdim=True)+1e-5)
        
        output = torch.matmul(proj_value, attention_matrix).view(n,-1,h,w)
        output = output
        
        return output

if __name__ == '__main__':
    ras          = PraNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out          = ras(input_tensor)