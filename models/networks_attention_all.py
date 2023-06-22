import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from einops import rearrange, repeat
from .CrossFusion import CrossFusionModule


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x.contiguous())) if i < self.num_layers - 1 else layer(x.contiguous())
        
        x = x.contiguous().permute(0,2,1).reshape([B,self.output_dim,H,W])
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5    
        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # reshape the context feature
        b, c, h, w = context.shape
        context = context.reshape(b, c, h*w).permute(0, 2, 1).contiguous()  # [b,c,h,w]->[b,n=(h*w),d=c]

        b, c, h, w = x.shape

        x = self.proj_in(x)   
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()   

        Q = self.Wq(x)  
        K = self.Wk(context)  
        V = self.Wv(context)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w).contiguous()   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        return out
    

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling
        #print self.feature_extraction

        if pool_type == 'conv1x1':
            self.conv1x1 = create_conv(512, 128, 1, 0)
            self.conv1x1.apply(weights_init)

            self.conv1x1_2 = create_conv(1024, 512, 1, 0)
            self.conv1x1_2.apply(weights_init)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

        # for position
        self.mlp = MLP(64, 256, 512, 2)
        # add attention
        self.attenion_visual = CrossAttention(in_channels=512, emb_dim=512)
        self.attenion_pos = CrossAttention(in_channels=512, emb_dim=512)
        # self.cross_attention = CrossFusionModule(hidden_dim=512, num_encoder_layers=1)

    def forward(self, x, y):
        x = self.feature_extraction(x)      # [512,7,7]

        # position feature
        y = self.mlp(y)     # out: [B C=512 H=7 W=7]

        # add cross attention
        x = x + self.attenion_visual(x, y)
        y = y + self.attenion_pos(y, x)
        
        z = torch.cat([x, y], dim=1)
        z = self.conv1x1_2(z.contiguous())

        return z

class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

        # cross attention
        self.cross_attention = CrossFusionModule(hidden_dim=512, num_encoder_layers=1)

    # 加在上采样的block中
    def forward(self, x, visual_feat_ori):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)       # [512,8,8]
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)       # [512,4,4]
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)       # [512,2,2]

        # 融合AV特征
        audio_multiscale = torch.cat([audio_conv5feature.flatten(2), audio_conv6feature.flatten(2), audio_conv7feature.flatten(2)], dim=2)
        audioVisual_feature = self.cross_attention(audio_multiscale, visual_feat_ori)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        return mask_prediction


class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction