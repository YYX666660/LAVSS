import torch
import torchvision
from .networks_pos_add import Resnet18, AudioVisual5layerUNet, AudioVisual7layerUNet, weights_init


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
            net.load_state_dict(torch.load(weights), strict=False)
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
            net.load_state_dict(torch.load(weights))
        return net

