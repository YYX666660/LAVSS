import torch
import torchvision
from .networks import Resnet18, AudioVisual5layerUNet, AudioVisual7layerUNet, weights_init


class ModelBuilder():
    # builder for visual stream
    def build_visual(self, pool_type='avgpool', input_channel=3, fc_out=512, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        if pool_type == 'conv1x1': #if use conv1x1, use conv1x1 + fc to reduce dimension to 512 feature vector
            net = Resnet18(original_resnet, pool_type=pool_type, input_channel=3, with_fc=True, fc_in=6272, fc_out=fc_out)
        else:
            net = Resnet18(original_resnet, pool_type=pool_type)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights), strict=True)
        return net

    #builder for audio stream
    def build_unet(self, unet_num_layers=7, ngf=64, input_nc=1, output_nc=1, weights=''):
        if unet_num_layers == 7:
            net = AudioVisual7layerUNet(ngf, input_nc, output_nc)
        elif unet_num_layers == 5:
            net = AudioVisual5layerUNet(ngf, input_nc, output_nc)

        net.apply(weights_init)

        if len(weights) > 0:
            print('Loading weights for UNet')
            # 加入部分层导入
            weight = torch.load(weights)
            part_weight = {k: v for k, v in weight.items() if k not in ['audionet_convlayer1.0.weight', 'audionet_convlayer1.0.bias', 'audionet_convlayer1.1.weight', 'audionet_convlayer1.1.bias', 'audionet_convlayer1.1.running_mean', 'audionet_convlayer1.1.running_var', 'audionet_convlayer1.1.num_batches_tracked']}
            net.load_state_dict(part_weight, strict=False)
        return net