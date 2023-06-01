# System libs
import os
import random
import time
   
os.environ["CUDA_VISIBLE_DEVICES"] = "6,5,4"

# Numerical libs
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io.wavfile as wavfile
from imageio import imsave
from mir_eval.separation import bss_eval_sources

# Our libs
from arguments_solo_phase import ArgParser
from dataset_solo_phase_loss import MUSICMixDataset 
from models.models import ModelBuilder      # add
from models.audioVisual_model import AudioVisualModel
from models import criterion
from utils import AverageMeter, \
    recover_rgb, magnitude2heatmap,\
    istft_reconstruction, istft_torch, stft_torch, warpgrid, \
    combine_video_audio, save_video, makedirs, extract_video, combine_audio
from viz import plot_loss_metrics, HTMLVisualizer
from torch.autograd import Variable


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_visual, self.net_unet = nets
        self.crit, self.crit_2 = crit

    def forward(self, batch_data, args):    # 关键主程序，送数据，对应music里的dict数据
        # left
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10   # [32,1,512,40]
        audio_mix_l = batch_data['audio_mix_l']

        # right
        mag_mix_r = batch_data['mag_mix_r']
        mags_r = batch_data['mags_r']
        mag_mix_r = mag_mix_r + 1e-10   # [32,1,512,40]
        audio_mix_r = batch_data['audio_mix_r']

        # phase diff
        IPD = batch_data['phase_diff']

        # position
        # position = batch_data['position']   # add

        N = args.num_mix    #2
        B = mag_mix.size(0)     #32
        T = mag_mix.size(3)     #256

        # 0.0 warp the spectrogram
        if args.log_freq:
            # left
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)     # 设置频谱图网格参数并转化为张量    [32,256,40,2]
            grid_warp_r = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)     # 采样数据，传入feature map,传出的张量大小[32，1，256，40]
            mag_mix_r = F.grid_sample(mag_mix_r, grid_warp_r)
            IPD = F.grid_sample(IPD, grid_warp)     # 采样数据，传入feature map,传出的张量大小[32，1，256，40]
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)
                mags_r[n] = F.grid_sample(mags_r[n], grid_warp_r)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)

            weight_r = torch.log1p(mag_mix_r)
            weight_r = torch.clamp(weight_r, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)   # 返回一个填充了标量值1的张量，其大小与之相同 input[32，1，256，256]
            weight_r = torch.ones_like(mag_mix_r)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        gt_masks_r = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
                gt_masks_r[n] = (mags_r[n] > 0.5 * mag_mix_r).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                gt_masks_r[n] = mags_r[n] / mag_mix_r
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)
                gt_masks_r[n].clamp_(0., 5.)

        # 2. forward net_frame -> Bx1xC (左右都一样)
        feat_frames = [None for n in range(N)]      # frame大小[32,32]
        for n in range(N):
            feat_frames[n] = self.net_visual(Variable(frames[n], requires_grad=False))
            # feat_frames[n], feat_frames_ori[n] = self.net_visual(Variable(frames[n], requires_grad=False), Variable(position[n], requires_grad=False))      # 有pos

        # 1. forward net_sound -> BxCxHxW
        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()   # 对数频谱，大小[32，1，256，256]
        log_mag_mix_r = torch.log(mag_mix_r).detach()

        # 3. sound synthesizer
        pred_masks = [None for n in range(N)]   # [32,1,256,256]
        pred_masks_r = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_unet(torch.cat([log_mag_mix, IPD], dim=1), feat_frames[n])
            pred_masks_r[n] = self.net_unet(torch.cat([log_mag_mix_r, IPD], dim=1), feat_frames[n])

        
        # 4. loss
        err = self.crit(pred_masks, gt_masks, weight).reshape(1)
        err_r = self.crit(pred_masks_r, gt_masks_r, weight_r).reshape(1)
        # add L2 loss
        err_2 = self.crit_2(pred_masks, gt_masks, weight).reshape(1)
        err_r_2 = self.crit_2(pred_masks_r, gt_masks_r, weight_r).reshape(1)
        # add time loss
        err_time = calc_time_loss(audio_mix_l, pred_masks, batch_data, args)
        err_time_r = calc_time_loss(audio_mix_r, pred_masks_r, batch_data, args)

        loss = (1/3) * (err + err_r) + (1/3) * (err_2 + err_r_2) + (1/3) * (err_time + err_time_r) 
        # loss = (1/3) * (err + err_r) + (1/3) * (err_2 + err_r_2) + (1/3) * (err_time + err_time_r)

        return err, err_r, loss, \
            {'pred_masks': pred_masks, 'gt_masks': gt_masks,
             'mag_mix': mag_mix, 'mags': mags, 'pred_masks_r': pred_masks_r, 'gt_masks_r': gt_masks_r,
             'mag_mix_r': mag_mix_r, 'mags_r': mags_r, 'weight': weight, 'weight_r': weight_r}


# 算时域loss
def calc_time_loss(audio_mix, mask, batch_data, args):
    # cal mix spec
    spec_mix = stft_torch(audio_mix, 1022, 256)

    # unwarp log scale
    N = args.num_mix
    B = spec_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, mask[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(mask[n], grid_unwarp)
        else:
            pred_masks_linear[n] = mask[n]

    pred_spec = torch.cat([spec_mix * pred_masks_linear[0], spec_mix * pred_masks_linear[1]], 1)
    preds = istft_torch(pred_spec, 1022, 256, seq_len=65535)
    audios = batch_data['audios']
    L = preds.shape[-1]
    target = torch.stack([audios[0][:, 0:L], audios[1][:, 0:L]], 1)     # [B, 2, 65535]

    # cal SI-SNR Loss
    # sisnr_loss = 0.
    # loss_list = torch.zeros(B)
    # for b in range(B):
    #     loss_0 = cal_SISNR(preds[b, 0], target[b, 0])
    #     loss_1 = cal_SISNR(preds[b, 1], target[b, 1])
    #     loss_list[b] = (loss_0 + loss_1) / 2

    # sisnr_loss = torch.mean(loss_list)

    # cal L1 time loss
    crit = criterion.L1Loss()
    sisnr_loss = crit(preds, target).reshape(1)
    return sisnr_loss

def cal_SISNR(preds, target):
    EPS = 1e-10
    target = target - torch.mean(target, dim=-1, keepdim=True)
    preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    alpha = (torch.sum(preds *target, dim=-1, keepdim=True) + EPS) / (torch.sum(target ** 2, dim=-1, keepdim=True) + EPS)
    target_scaled = alpha * target

    noise = target_scaled - preds
    val = (torch.sum(noise ** 2, dim=-1) + EPS) / (torch.sum(target_scaled ** 2, dim=-1) + EPS)
    val = 10 * torch.log10(val)
    loss = val
    return loss


# Calculate metrics  取两个
def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']
    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]

# Calculate metrics-right
def calc_metrics_r(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix_r']
    phase_mix = batch_data['phase_mix_r']
    audios = batch_data['audios_r']
    pred_masks_ = outputs['pred_masks_r']

    # unwarp log scale
    N = args.num_mix
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]

# Visualize predictions:将mask和audio输出并保存在目录下
def output_visuals(vis_rows, batch_data, outputs, args):
    # 1.fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    frames = batch_data['frames']
    infos = batch_data['infos']

    pred_masks_ = outputs['pred_masks']
    gt_masks_ = outputs['gt_masks']
    mag_mix_ = outputs['mag_mix']
    weight_ = outputs['weight']

    mag_mix_r = batch_data['mag_mix_r']
    phase_mix_r = batch_data['phase_mix_r']
    # frames = batch_data['frames']
    # infos = batch_data['infos']

    pred_masks_r = outputs['pred_masks_r']
    gt_masks_r = outputs['gt_masks_r']
    mag_mix_r_ = outputs['mag_mix_r']
    weight_r = outputs['weight_r']

    # 2.unwarp log scale  得到pred/gt的linear mask
    N = args.num_mix    # 2
    B = mag_mix.size(0)     # 32
    pred_masks_linear = [None for n in range(N)]    # [32,1,512,256]
    gt_masks_linear = [None for n in range(N)]      # [32,1,512,256]

    pred_masks_linear_r = [None for n in range(N)]
    gt_masks_linear_r = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
            gt_masks_linear[n] = F.grid_sample(gt_masks_[n], grid_unwarp)

            grid_unwarp_r = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, gt_masks_r[0].size(3), warp=False)).to(args.device)
            pred_masks_linear_r[n] = F.grid_sample(pred_masks_r[n], grid_unwarp_r)
            gt_masks_linear_r[n] = F.grid_sample(gt_masks_r[n], grid_unwarp_r)
        else:
            pred_masks_linear[n] = pred_masks_[n]
            gt_masks_linear[n] = gt_masks_[n]

            pred_masks_linear_r[n] = pred_masks_r[n]
            gt_masks_linear_r[n] = gt_masks_r[n]

    # 3.convert into numpy
    mag_mix = mag_mix.numpy()       # [32,1,512,256]
    mag_mix_ = mag_mix_.detach().cpu().numpy()      # [32,1,256,256]
    phase_mix = phase_mix.numpy()        # [32,1,512,256]
    weight_ = weight_.detach().cpu().numpy()        # [32,1,256,256]
    # right
    mag_mix_r = mag_mix_r.numpy()       # [32,1,512,256]
    mag_mix_r_ = mag_mix_r_.detach().cpu().numpy()      # [32,1,256,256]
    phase_mix_r = phase_mix_r.numpy()        # [32,1,512,256]
    weight_r = weight_r.detach().cpu().numpy()        # [32,1,256,256]
    for n in range(N):
        pred_masks_[n] = pred_masks_[n].detach().cpu().numpy()
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()
        gt_masks_[n] = gt_masks_[n].detach().cpu().numpy()
        gt_masks_linear[n] = gt_masks_linear[n].detach().cpu().numpy()

        pred_masks_r[n] = pred_masks_r[n].detach().cpu().numpy()
        pred_masks_linear_r[n] = pred_masks_linear_r[n].detach().cpu().numpy()
        gt_masks_r[n] = gt_masks_r[n].detach().cpu().numpy()
        gt_masks_linear_r[n] = gt_masks_linear_r[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_[n] = (pred_masks_[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

            pred_masks_r[n] = (pred_masks_r[n] > args.mask_thres).astype(np.float32)
            pred_masks_linear_r[n] = (pred_masks_linear_r[n] > args.mask_thres).astype(np.float32)

    # 4.loop over each sample
    for j in range(B):
        row_elements = []

        # video names
        prefix = []
        for n in range(N):
            prefix.append('-'.join(infos[n][0][j].split('/')[-2:]).split('.')[0])
        prefix = '+'.join(prefix)
        makedirs(os.path.join(args.vis, prefix))

        # 4.1 save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)        # 65280
        mix_amp = magnitude2heatmap(mag_mix_[j, 0])     # [256,256,3]
        weight = magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(args.vis, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(args.vis, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav), args.audRate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        mix_wav_r = istft_reconstruction(mag_mix_r[j, 0], phase_mix_r[j, 0], hop_length=args.stft_hop)        # 65280
        mix_amp_r = magnitude2heatmap(mag_mix_r_[j, 0])     # [256,256,3]
        weightr = magnitude2heatmap(weight_r[j, 0], log=False, scale=100.)
        filename_mixwav_r = os.path.join(prefix, 'mix_r.wav')
        filename_mixmag_r = os.path.join(prefix, 'mix_r.jpg')
        filename_weight_r = os.path.join(prefix, 'weight_r.jpg')
        imsave(os.path.join(args.vis, filename_mixmag_r), mix_amp_r[::-1, :, :])
        imsave(os.path.join(args.vis, filename_weight_r), weightr[::-1, :])
        wavfile.write(os.path.join(args.vis, filename_mixwav_r), args.audRate, mix_wav_r)
        row_elements += [{'text': prefix}, {'image': filename_mixmag_r, 'audio': filename_mixwav_r}]

        # 4.2 save each component
        preds_wav = [None for n in range(N)]
        preds_wav_r = [None for n in range(N)]
        for n in range(N):
            # 4.2.1 GT and predicted audio recovery
            gt_mag = mag_mix[j, 0] * gt_masks_linear[n][j, 0]       # [512,256]
            gt_wav = istft_reconstruction(gt_mag, phase_mix[j, 0], hop_length=args.stft_hop)    # 65280
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]       # [512,256]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)     # 65280

            gt_mag_r = mag_mix_r[j, 0] * gt_masks_linear_r[n][j, 0]       # [512,256]
            gt_wav_r = istft_reconstruction(gt_mag_r, phase_mix_r[j, 0], hop_length=args.stft_hop)    # 65280
            pred_mag_r = mag_mix_r[j, 0] * pred_masks_linear_r[n][j, 0]       # [512,256]
            preds_wav_r[n] = istft_reconstruction(pred_mag_r, phase_mix_r[j, 0], hop_length=args.stft_hop)     # 65280

            # 4.2.2 output masks
            filename_gtmask = os.path.join(prefix, 'gtmask{}.jpg'.format(n+1))
            filename_predmask = os.path.join(prefix, 'predmask{}.jpg'.format(n+1))
            gt_mask = (np.clip(gt_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask = (np.clip(pred_masks_[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imsave(os.path.join(args.vis, filename_gtmask), gt_mask[::-1, :])
            imsave(os.path.join(args.vis, filename_predmask), pred_mask[::-1, :])

            filename_gtmask_r = os.path.join(prefix, 'gtmask_r{}.jpg'.format(n+1))
            filename_predmask_r = os.path.join(prefix, 'predmask_r{}.jpg'.format(n+1))
            gt_mask_r = (np.clip(gt_masks_r[n][j, 0], 0, 1) * 255).astype(np.uint8)
            pred_mask_r = (np.clip(pred_masks_r[n][j, 0], 0, 1) * 255).astype(np.uint8)
            imsave(os.path.join(args.vis, filename_gtmask_r), gt_mask_r[::-1, :])
            imsave(os.path.join(args.vis, filename_predmask_r), pred_mask_r[::-1, :])

            # 4.2.3 ouput spectrogram (log of magnitude, show colormap)
            filename_gtmag = os.path.join(prefix, 'gtamp{}.jpg'.format(n+1))
            filename_predmag = os.path.join(prefix, 'predamp{}.jpg'.format(n+1))
            gt_mag = magnitude2heatmap(gt_mag)      # 频谱图转换为热图
            pred_mag = magnitude2heatmap(pred_mag)
            imsave(os.path.join(args.vis, filename_gtmag), gt_mag[::-1, :, :])
            imsave(os.path.join(args.vis, filename_predmag), pred_mag[::-1, :, :])

            filename_gtmag_r = os.path.join(prefix, 'gtamp_r{}.jpg'.format(n+1))
            filename_predmag_r = os.path.join(prefix, 'predamp_r{}.jpg'.format(n+1))
            gt_mag_r = magnitude2heatmap(gt_mag_r)      # 频谱图转换为热图
            pred_mag_r = magnitude2heatmap(pred_mag_r)
            imsave(os.path.join(args.vis, filename_gtmag_r), gt_mag_r[::-1, :, :])
            imsave(os.path.join(args.vis, filename_predmag_r), pred_mag_r[::-1, :, :])

            # 4.2.4 output audio
            filename_gtwav = os.path.join(prefix, 'gt{}.wav'.format(n+1))
            filename_predwav = os.path.join(prefix, 'pred{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav), args.audRate, gt_wav)
            wavfile.write(os.path.join(args.vis, filename_predwav), args.audRate, preds_wav[n])

            filename_gtwav_r = os.path.join(prefix, 'gt_r{}.wav'.format(n+1))
            filename_predwav_r = os.path.join(prefix, 'pred_r{}.wav'.format(n+1))
            wavfile.write(os.path.join(args.vis, filename_gtwav_r), args.audRate, gt_wav_r)
            wavfile.write(os.path.join(args.vis, filename_predwav_r), args.audRate, preds_wav_r[n])

            # 4.2.5 output video
            path_video = os.path.join(args.vis, prefix, 'video{}.mp4'.format(n+1))
            item = prefix.split('+')[n].split('-')[1] + '.mp4'
            ori_video = os.path.join('/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/videos',item)
            extract_video(ori_video, path_video)
           
            # 4.2.6 combine pred video and audio
            # combine two channels audios
            filename_combine_predwav = os.path.join(prefix, 'stereo{}.wav'.format(n+1))
            combine_audio(
                os.path.join(args.vis, filename_predwav), 
                os.path.join(args.vis, filename_predwav_r), 
                os.path.join(args.vis, filename_combine_predwav))

            filename_av = os.path.join(prefix, 'av{}.mp4'.format(n+1))
            combine_video_audio(
                path_video,
                os.path.join(args.vis, filename_combine_predwav),
                os.path.join(args.vis, filename_av))

            row_elements += [
                {'video': filename_av},
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    loss_meter_l = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    loss_meter_r = AverageMeter()
    sdr_mix_meter_r = AverageMeter()
    sdr_meter_r = AverageMeter()
    sir_meter_r = AverageMeter()
    sar_meter_r = AverageMeter()

    # initialize HTML header
    visualizer = HTMLVisualizer(os.path.join(args.vis, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    for n in range(1, args.num_mix+1):
        header += ['Video {:d}'.format(n),
                   'Predicted Audio {:d}'.format(n),
                   'GroundTruth Audio {}'.format(n),
                   'Predicted Mask {}'.format(n),
                   'GroundTruth Mask {}'.format(n)]
    header += ['Loss weighting']
    visualizer.add_header(header)
    vis_rows = []

    for i, batch_data in enumerate(loader):
        # forward pass: err err_r loss
        err, err_r, loss, outputs = netWrapper.forward(batch_data, args)
        loss = loss.mean()
        err = err.mean()
        err_r = err_r.mean()

        loss_meter.update(loss.item())
        loss_meter_l.update(err.item())
        loss_meter_r.update(err_r.item())
        print('[Eval] iter {}, loss: {:.4f}, loss_l: {:.4f}, loss_r: {:.4f}'.format(i, loss.item(), err.item(), err_r.item()))

        # calculate metrics
        sdr_mix, sdr, sir, sar = calc_metrics(batch_data, outputs, args)
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        sdr_mix_r, sdr_r, sir_r, sar_r = calc_metrics_r(batch_data, outputs, args)
        sdr_mix_meter_r.update(sdr_mix_r)
        sdr_meter_r.update(sdr_r)
        sir_meter_r.update(sir_r)
        sar_meter_r.update(sar_r)

        # output visualization
        # if len(vis_rows) < args.num_vis:
            # output_visuals(vis_rows, batch_data, outputs, args)

    print('[Eval Summary] Epoch: {}, Loss: {:.4f}, Loss_l: {:.4f}, Loss_r: {:.4f} '
          'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f} '
          'SDR_mixture_r: {:.4f}, SDR_r: {:.4f}, SIR_r: {:.4f}, SAR_r: {:.4f} '
          .format(epoch, loss_meter.average(), loss_meter_l.average(), loss_meter_r.average(),
                  sdr_mix_meter.average(),
                  sdr_meter.average(),
                  sir_meter.average(),
                  sar_meter.average(),
                  sdr_mix_meter_r.average(),
                  sdr_meter_r.average(),
                  sir_meter_r.average(),
                  sar_meter_r.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    print('Plotting html for visualization...')
    visualizer.add_rows(vis_rows)
    visualizer.write_html()

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err, err_r, loss, _ = netWrapper.forward(batch_data, args)
        loss = loss.mean()

        # backward  
        loss.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_unet: {}, lr_visual: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_unet, args.lr_visual,
                          loss.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(loss.item())


def checkpoint(nets, history, epoch, args):     # 保存每一epoch的模型
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_frame, net_sound) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'
    suffix_sdrbest = 'sdrbest.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    

    cur_err = history['val']['err'][-1]         # 这里应该是总的loss才对,需要改一下
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))

    cur_sdr = history['val']['sdr'][-1]   
    if cur_sdr > args.best_sdr:
        args.best_sdr = cur_sdr
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_sdrbest))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_sdrbest))

def create_optimizer(nets, args):
    (net_visual, net_unet) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': args.lr_visual},
                    {'params': net_unet.parameters(), 'lr': args.lr_unet}]
    return torch.optim.Adam(param_groups, betas=(args.beta1,0.999), weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    
    args.lr_visual *= 0.1
    args.lr_unet *= 0.1


def main(args):
    # 1. Network Builders
    builder = ModelBuilder()
    net_visual = builder.build_visual(
            pool_type=args.visual_pool,
            fc_out = 512,
            weights=args.weights_visual)  
    net_unet = builder.build_unet(
            unet_num_layers = args.unet_num_layers,
            ngf=args.unet_ngf,
            input_nc=args.unet_input_nc,
            output_nc=args.unet_output_nc,
            weights=args.weights_unet)
    # net_classifier = builder.build_classifier(
    #         pool_type=opt.classifier_pool,
    #         num_of_classes=opt.number_of_classes,
    #         input_channel=opt.unet_output_nc,
    #         weights=opt.weights_classifier)
    nets = (net_visual, net_unet)
    # print(nets)
    crit = criterion.L1Loss()
    crit_2 = criterion.L2Loss()
    
    print('Fnished Network Builders')

    # 2. Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split='val')

    loader_train = torch.utils.data.DataLoader(     # len = 14960
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(       # len = 375
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))
    
    # 3. Wrap networks:三个网络 L1标准
    netWrapper = NetWrapper(nets, [crit, crit_2])
    device_id = [0,1,2] 
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=device_id)      # 任意选gpu跑,[0,1]对应开头的["CUDA_VISIBLE_DEVICES"] = "3, 4"
    netWrapper.to(args.device)
    
    print('finished wrapping')

    # 4. Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'sdr': [], 'sir': [], 'sar': []}}

    # Eval mode
    # evaluate(netWrapper, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            # checkpointing
            checkpoint(nets, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)
        # if(args.learning_rate_decrease_itr > 0 and epoch % args.learning_rate_decrease_itr == 0):
        #     adjust_learning_rate(optimizer, args.decay_factor)
        #     print('decreased learning rate by ', args.decay_factor)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            # assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization_test/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.weights_unet = os.path.join(args.ckpt, 'sound_sdrbest.pth')
        args.weights_visual = os.path.join(args.ckpt, 'frame_sdrbest.pth')
        # args.weights_synthesizer = os.path.join(args.ckpt, 'synthesizer_latest.pth')

    # initialize best error with a big number
    args.best_err = float("inf")
    args.best_sdr = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)    #为CPU设置种子用于生成随机数，以使得结果是确定的
    main(args)
