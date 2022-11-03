import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
# from torchvision.transforms import InterpolationMode    # added
# import sys
# sys.path.append('/home/workspace/weitang/anaconda3/envs/yyx1/lib/python3.6/site-packages/')
# import torchaudio
import librosa
from PIL import Image 

from . import video_transforms as vtransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        # load pos_npy
        self.whole_img_position = np.load("whole_img_position_64.npy")

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # initialize image transform
        self._init_transform()
        
        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            # self.list_sample = [x.rstrip() for x in open(list_sample, 'r')]
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.2)),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        # frames = []
        # for path in paths:
        #     frames.append(self._load_frame(path))
        # frames = self.vid_transform(frames)
        frames = self._load_frame(paths[0])
        frames = self.img_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img    

    ##### 2、取整个的frame+一小块的position
    def _load_frames_det(self, paths, path_frames_ids,  path_frames_det):
        det_res = np.load(path_frames_det)
        position = []

        # frame
        frames = self._load_frame(paths[1])
        frames = self.img_transform(frames)

        # position
        pos = self._load_position_det(paths[1], 1, det_res)
        position = self.transform_pos(pos)
        

        return frames, position

    def _load_frame_det(self, path, id, det_res):

        # load image
        img = Image.open(path).convert('RGB')

        bb = det_res[id, 3:]
        # crop image
        img = img.crop((bb[0], bb[1], bb[2], bb[3]))    # 此处把bounding box的区域切取出来了
        
        return img
    
    def transform_pos(self, pos):
        pos = torch.from_numpy(pos)
        pos = pos.unsqueeze(0)   # [1,W,H,C]
        pos = pos.permute(0, 3, 2, 1)   # [N,C,H,W]
        # pos = torch.nn.functional.interpolate(pos, size=[224,224], mode='bilinear', align_corners = False)      # [1,c,224,224]
        pos = torch.nn.functional.adaptive_max_pool2d(pos, 7)       # [1,c,7,7]
        pos = pos.float()
        pos = pos.squeeze()     # [C,H,W]
        return pos
    
    def _load_position_det(self, path, id, det_res):

        # load npy
        # whole_img_position = self.whole_img_position

        bb = det_res[id, 3:]
        bb = bb.astype("int")

        # positional encoding
        position_matrix1 = self.whole_img_position[bb[0]:bb[2], bb[1]:bb[3], :]
        # position_matrix1 = self._get_position(bb[0], bb[1], bb[2], bb[3])

        return position_matrix1
    
    def _get_position(self, left, up, right, below):

        H, W = int(below) - int(up), int(right) - int(left)
    
        position_matrix = [[[] for x in range(H)] for y in range(W)]     # H*W*40

        L = 10

        x_linspace = ((np.linspace(left, right-1, W) - left)/W)*2 -1    # 归一化到【-1，1】之间
        y_linspace = ((np.linspace(up, below-1, H) - up)/H)*2 -1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []
        
        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el 
            
            x = np.sin(val * np.pi * x_linspace)
            x_el.append(x)      # 10,256

            x = np.cos(val * np.pi * x_linspace)
            x_el_hf.append(x)

            y = np.sin(val * np.pi * y_linspace)
            y_el.append(y)

            y = np.cos(val * np.pi * y_linspace)
            y_el_hf.append(y)

        # TODO: vectorise this code!
        for x_i in range(0, W):
            for y_i in range(0, H):

                # r, g, b = image_np[y_i, x_i]      # 需要rgb信息吗？

                p_enc = []      # 40

                for li in range(0, L):

                    p_enc.append(x_el[li][x_i])
                    p_enc.append(x_el_hf[li][x_i])

                    p_enc.append(y_el[li][y_i])
                    p_enc.append(y_el_hf[li][y_i])

                # p_enc = p_enc + [x_i, y_i, r*2 -1, g*2 -1, b*2 -1]      # 45

                position_matrix[x_i][y_i]  = p_enc
        
        position_matrix = np.asarray(position_matrix)
        

        return position_matrix

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)     # [512,256]
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        # if path.endswith('.mp3'):
        #     audio_raw, rate = torchaudio.load(path)
        #     audio_raw = audio_raw.numpy().astype(np.float32)

        #     # range to [-1, 1]
        #     audio_raw *= (2.0**-31)

        #     # convert to mono
        #     # if audio_raw.shape[1] == 2:
        #     #     audio_raw = (audio_raw[:, 0] + audio_raw[:, 1]) / 2
        #     # else:
        #     #     audio_raw = audio_raw[:, 0]
        # else:
        audio_raw, rate = librosa.load(path, sr=self.audRate, mono=False)


        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio_l = np.zeros(self.audLen, dtype=np.float32)
        audio_r = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio_l

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short (no)
        if audio_raw.shape[1] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample  (no)
        if rate > self.audRate:
            # print('resmaple {}->{}'.format(rate, self.audRate))
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[1]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)
        audio_raw_all = audio_raw[:, start:end]

        # 获取左声道
        audio_l[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw_all[0,:]      

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio_l *= scale
        audio_l[audio_l > 1.] = 1.
        audio_l[audio_l < -1.] = -1.

        # 获取右声道
        audio_r[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw_all[1,:]      

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio_r *= scale
        audio_r[audio_r > 1.] = 1.
        audio_r[audio_r < -1.] = -1.

        return audio_l, audio_r

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # mix
        for n in range(N):
            audios[n] /= N
        audio_mix = np.asarray(audios).sum(axis=0)

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        # to tensor
        # audio_mix = torch.from_numpy(audio_mix)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)   # amp_mix混合幅值，mags每种乐器单独的幅值,phase_mix混合相位

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)

        return amp_mix, mags, frames, audios, phase_mix
