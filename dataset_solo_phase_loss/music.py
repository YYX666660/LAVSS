# 3、目标检测后的小块frames+position
import os
import random
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix
        # print(list_sample)

    def __getitem__(self, index):
        N = self.num_mix    # mix的乐器种类数
        frames = [None for n in range(N)]
        audios_l = [None for n in range(N)]
        audios_r = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_frames_ids = [[] for n in range(N)]
        path_frames_det = ['' for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        position = [None for n in range(N)]
        class_list = []

        # the first video
        infos[0] = self.list_sample[index]
        # cls = infos[0][3]  
        # class_list.append(cls)      # 获取第一个info的类别
        # print(infos)

        # sample other videos:for train（不带乐器类别判断）
        if not self.split == 'train':
            random.seed(index)
        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]

        # sample other videos:for test（带乐器类别判断）
        # if not self.split == 'train':
        #     random.seed(index)
        # for n in range(1, N):
        #     indexN = random.randint(0, len(self.list_sample)-1)
        #     sample = self.list_sample[indexN]
        #     while sample[3] in class_list:    # 如果已经有该类别了，就重新选其他类别，确保选取的两个info类别不重复！
        #         indexN = random.randint(0, len(self.list_sample) - 1)
        #         sample = self.list_sample[indexN]
        #     infos[n] = sample
        #     class_list.append(sample[3])      # 有2个值

        # select frames 对frames进行操作
        idx_margin = max(
            int(self.fps * 8), (self.num_frames // 2) * self.stride_frames)     # 64
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_framesN = infoN

            path_audioN = '/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration' + path_audioN.split('.')[1] + '.wav'
            path_frameN = '/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration' + path_frameN.split('.')[1] + '.mp4'

            # 试一试 无论train还是eval都取中间值41吧
            # if self.split == 'train':
            #     # random, not to sample start and end n-frames
            #     center_frameN = random.randint(
            #         idx_margin+1, int(count_framesN)-idx_margin)
            # else:
            center_frameN = int(count_framesN) // 2     # 82/2=41
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames    # 0
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))   # 选取接近的中心+偏移的帧图片 选了第41帧，总共有82帧
                path_frames_ids[n].append(center_frameN + idx_offset)       # 记录帧的id
            path_frames_det[n] = os.path.join("/home/yeyx/Data/Audio-Visual-Spatial-Audio-Sepration/data/detection_results",
                        path_frameN.split('/')[-1]+'.npy')

            path_audios[n] = path_audioN

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])     # 包含整块的frame
                # frames[n], position[n] = self._load_frames_det(path_frames[n], path_frames_ids[n], path_frames_det[n])   # 包含frame+position
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps      # 5.0625
                audios_l[n], audios_r[n]= self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix, spec_mix_l, audio_mix_l = self._mix_n_and_stft(audios_l)
            mag_mix_r, mags_r, phase_mix_r, spec_mix_r, audio_mix_r = self._mix_n_and_stft(audios_r)

            # cal phase diff
            import numpy as np
            import torch
            EPS = 1e-10
            phase_diff = spec_mix_l * spec_mix_r.conj() / ((np.abs(spec_mix_l) + EPS) * (np.abs(spec_mix_r) + EPS))
            phase_diff = torch.from_numpy(np.real(phase_diff)).unsqueeze(0)

        except Exception as e:      # 错误明细检查
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, audios_l, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'mag_mix_r': mag_mix_r, 'mags_r': mags_r, 'phase_diff': phase_diff, 
                    'audio_mix_l': audio_mix_l, 'audio_mix_r': audio_mix_r}
        # ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags, 'mag_mix_r': mag_mix_r, 'mags_r': mags_r, 'position': position, 'phase_diff': phase_diff}
        # if self.split != 'train':
        ret_dict['audios'] = audios_l
        ret_dict['phase_mix'] = phase_mix
        ret_dict['audios_r'] = audios_r
        ret_dict['phase_mix_r'] = phase_mix_r
        ret_dict['infos'] = infos

        return ret_dict