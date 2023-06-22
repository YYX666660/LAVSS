##### for train #####
import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Model related arguments
        parser.add_argument('--id', default='withpos_sq_cross_attention_then_concat_MLP_finetune_5e-5_1e-4',                         # 改
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="number of sounds to mix")
        parser.add_argument('--arch_sound', default='unet7',
                            help="architecture of net_sound")
        parser.add_argument('--arch_frame', default='resnet18dilated',
                            help="architecture of net_frame")
        parser.add_argument('--arch_synthesizer', default='linear',
                            help="architecture of net_synthesizer")
        parser.add_argument('--weights_sound', default='',
                            help="weights to finetune net_sound")
        parser.add_argument('--weights_frame', default='',
                            help="weights to finetune net_frame")
        parser.add_argument('--weights_synthesizer', default='',
                            help="weights to finetune net_synthesizer")

        parser.add_argument('--num_channels', default=32, type=int,
                            help='number of channels')
        parser.add_argument('--num_frames', default=3, type=int,        # 改
                            help='number of frames')
        parser.add_argument('--stride_frames', default=20, type=int,        # 改
                            help='sampling stride of frames')
        parser.add_argument('--img_pool', default='maxpool',
                            help="avg or max pool image features")
        parser.add_argument('--img_activation', default='sigmoid',
                            help="activation on the image features")
        parser.add_argument('--sound_activation', default='no',
                            help="activation on the sound features")
        parser.add_argument('--output_activation', default='sigmoid',
                            help="activation on the output")
        parser.add_argument('--binary_mask', default=0, type=int,
                            help="whether to use bianry masks")
        parser.add_argument('--mask_thres', default=0.5, type=float,
                            help="threshold in the case of binary masks")
        parser.add_argument('--loss', default='l1',        # 改 原来l1
                            help="loss function to use")
        parser.add_argument('--weighted_loss', default=1, type=int,     # 改
                            help="weighted loss")
        parser.add_argument('--log_freq', default=1, type=int,
                            help="log frequency scale")
        
        parser.add_argument('--visual_pool', type=str, default='conv1x1', help='avg/max pool or using a conv1x1 layer for visual stream feature')
        parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")
        parser.add_argument('--weights_visual', type=str, default='v2_ckpt_attention/withoutpos_sq_cross_attention_then_concat_5e-4_musicpre-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch60-step40_60/frame_best.pth', help="weights for visual stream")
        parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")
        parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
        parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
        parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
        parser.add_argument('--weights_unet', type=str, default='v2_ckpt_attention/withoutpos_sq_cross_attention_then_concat_5e-4_musicpre-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch60-step40_60/sound_best.pth', help="weights for unet")
        parser.add_argument('--number_of_classes', default=15, type=int, help='number of classes')
        
        # Data related arguments
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--gpu_ids', type=str, default='0,1',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=12, type=int,      # 改 原32 现12，不能开太大了，否则会占太多CPU
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,     # 改 原-1 现256
                            help='number of images to evalutate')
        parser.add_argument('--num_vis', default=20, type=int,
                            help='number of images to evalutate')

        parser.add_argument('--audLen', default=65535, type=int,
                            help='sound length')
        parser.add_argument('--audRate', default=11025, type=int,
                            help='sound sampling rate')
        parser.add_argument('--stft_frame', default=1022, type=int,
                            help="stft frame length")
        parser.add_argument('--stft_hop', default=256, type=int,
                            help="stft hop length")

        parser.add_argument('--imgSize', default=224, type=int,
                            help='size of input frame')
        parser.add_argument('--frameRate', default=8, type=float,
                            help='video frame sampling rate')

        # Misc arguments
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./v2_ckpt_attention',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=10,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_train.csv')
        parser.add_argument('--list_val',
                            default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_val.csv')
        parser.add_argument('--dup_trainset', default=10, type=int,
                            help='duplicate so that one epoch has more iters')

        # optimization related arguments
        parser.add_argument('--num_epoch', default=60, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')        # 改 原来是1e-4
        parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')         # 改 原来是1e-3
        parser.add_argument('--lr_synthesizer',
                            default=1e-3, type=float, help='LR')

        parser.add_argument('--lr_visual', type=float, default=5e-5, help='learning rate for visual stream')
        parser.add_argument('--lr_unet', type=float, default=1e-4, help='learning rate for unet')

        parser.add_argument('--lr_steps',
                            nargs='+', type=int, default=[15, 30],
                            help='steps to drop LR in epochs')          # 改 原来是[40,60]
        parser.add_argument('--learning_rate_decrease_itr', type=int, default=5, help='how often is the learning rate decreased by six percent')
        parser.add_argument('--decay_factor', type=float, default=0.9, help='learning rate decay factor')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weights regularizer')
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args





####### for test: use test_solo.csv  #########
# import argparse


# class ArgParser(object):
#     def __init__(self):
#         parser = argparse.ArgumentParser()
#         # Model related arguments
#         parser.add_argument('--id', default='withpos_sq_cross_attention_then_concat_MLP_finetune_5e-5_1e-4-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch60-step15_30',                         # 改
#                             help="a name for identifying the model")
#         parser.add_argument('--num_mix', default=2, type=int,
#                             help="number of sounds to mix")
#         parser.add_argument('--arch_sound', default='unet7',
#                             help="architecture of net_sound")
#         parser.add_argument('--arch_frame', default='resnet18dilated',
#                             help="architecture of net_frame")
#         parser.add_argument('--arch_synthesizer', default='linear',
#                             help="architecture of net_synthesizer")
#         parser.add_argument('--weights_sound', default='',
#                             help="weights to finetune net_sound")
#         parser.add_argument('--weights_frame', default='',
#                             help="weights to finetune net_frame")
#         parser.add_argument('--weights_synthesizer', default='',
#                             help="weights to finetune net_synthesizer")

#         parser.add_argument('--num_channels', default=32, type=int,
#                             help='number of channels')
#         parser.add_argument('--num_frames', default=3, type=int,        # 改
#                             help='number of frames')
#         parser.add_argument('--stride_frames', default=20, type=int,        # 改
#                             help='sampling stride of frames')
#         parser.add_argument('--img_pool', default='maxpool',
#                             help="avg or max pool image features")
#         parser.add_argument('--img_activation', default='sigmoid',
#                             help="activation on the image features")
#         parser.add_argument('--sound_activation', default='no',
#                             help="activation on the sound features")
#         parser.add_argument('--output_activation', default='sigmoid',
#                             help="activation on the output")
#         parser.add_argument('--binary_mask', default=0, type=int,
#                             help="whether to use bianry masks")
#         parser.add_argument('--mask_thres', default=0.5, type=float,
#                             help="threshold in the case of binary masks")
#         parser.add_argument('--loss', default='l1',        # 改 原来l1
#                             help="loss function to use")
#         parser.add_argument('--weighted_loss', default=1, type=int,     # 改
#                             help="weighted loss")
#         parser.add_argument('--log_freq', default=1, type=int,
#                             help="log frequency scale")
        
#         parser.add_argument('--visual_pool', type=str, default='conv1x1', help='avg/max pool or using a conv1x1 layer for visual stream feature')
#         parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")
#         parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
#         parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")
#         parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
#         parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
#         parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
#         parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
#         parser.add_argument('--weights_classifier', type=str, default='', help="weights for audio classifier")
#         parser.add_argument('--number_of_classes', default=15, type=int, help='number of classes')
        
#         # Data related arguments
#         parser.add_argument('--num_gpus', default=1, type=int,
#                             help='number of gpus to use')
#         parser.add_argument('--gpu_ids', type=str, default='0,1',
#                             help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#         parser.add_argument('--batch_size_per_gpu', default=64, type=int,
#                             help='input batch size')
#         parser.add_argument('--workers', default=16, type=int,      # 改 原32 现12，不能开太大了，否则会占太多CPU
#                             help='number of data loading workers')
#         parser.add_argument('--num_val', default=-1, type=int,     # 改 原-1 现256
#                             help='number of images to evalutate')
#         parser.add_argument('--num_vis', default=30, type=int,
#                             help='number of images to evalutate')

#         parser.add_argument('--audLen', default=65535, type=int,
#                             help='sound length')
#         parser.add_argument('--audRate', default=11025, type=int,
#                             help='sound sampling rate')
#         parser.add_argument('--stft_frame', default=1022, type=int,
#                             help="stft frame length")
#         parser.add_argument('--stft_hop', default=256, type=int,
#                             help="stft hop length")

#         parser.add_argument('--imgSize', default=224, type=int,
#                             help='size of input frame')
#         parser.add_argument('--frameRate', default=8, type=float,
#                             help='video frame sampling rate')

#         # Misc arguments
#         parser.add_argument('--seed', default=1234, type=int,
#                             help='manual seed')
#         parser.add_argument('--ckpt', default='./v2_ckpt_attention',
#                             help='folder to output checkpoints')
#         parser.add_argument('--disp_iter', type=int, default=10,
#                             help='frequency to display')
#         parser.add_argument('--eval_epoch', type=int, default=1,
#                             help='frequency to evaluate')

#         self.parser = parser

#     def add_train_arguments(self):
#         parser = self.parser

#         parser.add_argument('--mode', default='eval',
#                             help="train/eval")
#         parser.add_argument('--list_train',
#                             default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_train.csv')
#         parser.add_argument('--list_val',
#                             default='/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/solo_test.csv')
#         parser.add_argument('--dup_trainset', default=10, type=int,
#                             help='duplicate so that one epoch has more iters')

#         # optimization related arguments
#         parser.add_argument('--num_epoch', default=60, type=int,
#                             help='epochs to train for')
#         parser.add_argument('--lr_frame', default=5e-4, type=float, help='LR')        # 改 原来是1e-4
#         parser.add_argument('--lr_sound', default=5e-3, type=float, help='LR')         # 改 原来是1e-3
#         parser.add_argument('--lr_synthesizer',
#                             default=5e-3, type=float, help='LR')

#         parser.add_argument('--lr_visual', type=float, default=1e-4, help='learning rate for visual stream')
#         parser.add_argument('--lr_unet', type=float, default=1e-3, help='learning rate for unet')
#         parser.add_argument('--lr_classifier', type=float, default=0.001, help='learning rate for audio classifier')

#         parser.add_argument('--lr_steps',
#                             nargs='+', type=int, default=[40, 50],
#                             help='steps to drop LR in epochs')          # 改 原来是[40,60]
#         parser.add_argument('--beta1', default=0.9, type=float,
#                             help='momentum for sgd, beta1 for adam')
#         parser.add_argument('--weight_decay', default=1e-4, type=float,
#                             help='weights regularizer')
#         self.parser = parser

#     def print_arguments(self, args):
#         print("Input arguments:")
#         for key, val in vars(args).items():
#             print("{:16} {}".format(key, val))

#     def parse_train_arguments(self):
#         self.add_train_arguments()
#         args = self.parser.parse_args()
#         self.print_arguments(args)
#         return args