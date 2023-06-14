import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_transformer import CrossTransformerLayer,CrossTransformerEncoder


class CrossFusionModule(nn.Module):
    def __init__(self, hidden_dim=512, num_encoder_layers=1):
        super(CrossFusionModule, self).__init__()
        
        # encoder transformer 
        encoder_layer = CrossTransformerLayer(d_model = hidden_dim, nhead=8)
        self.encoder = CrossTransformerEncoder(encoder_layer, num_encoder_layers)

        #         
        self.norm_Fusion_A = nn.LayerNorm(hidden_dim)
        self.norm_Fusion_B = nn.LayerNorm(hidden_dim)
        
        self.conv_after_body_Fusion = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) # relu的一种变体

        self.proj_out1 = nn.Conv1d(49, 336, kernel_size=3, stride=1, padding=1)
        self.proj_out2 = nn.Conv1d(336, 256, kernel_size=5, stride=2, padding=2)
        
        
    def forward(self, x, y):
        # 首先将 x, y 展平
        # c = x.shape[1]
        # h = x.shape[2]
        # w = x.shape[3]
        x_flatten = x.permute(0, 2, 1)
        y_flatten = y.flatten(2).permute(0, 2, 1)
        
        # 经过一层cross attention 
        output_x, output_y = self.encoder(x_flatten, y_flatten, pos = None)
        
        # [b,h*w,c]->[b,c,h,w]
        # 拼接
        output_y = self.lrelu(self.proj_out1(output_y))
        x = torch.cat([output_x, output_y],2)
        x = self.lrelu(self.proj_out2(x))

        x = x.permute(0,2,1).reshape([-1,512,16,16])
                
        return x
        
    