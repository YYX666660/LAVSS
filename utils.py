import os
import shutil

import numpy as np
import librosa
import cv2

import subprocess as sp
from threading import Timer

import torch

# 设定网格参数在[-1,1]之间
def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)      # 强制转换
    return grid


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


def recover_rgb(img):
    for t, m, s in zip(img,
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.):
    if log:
        mag = np.log10(mag + 1.)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = mag_color[:, :, ::-1]
    return mag_color

# 幅值相位重建
def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)

# 实部虚部重建
def istft_reconstruction_(real, imag, hop_length=256):
    spec = real + 1j * imag
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)

def abs_phase_convert(mag, phase):
    real = mag * torch.cos(phase)
    img = mag * torch.sin(phase)
    spectro = torch.cat((real.unsqueeze(1), img.unsqueeze(1)), dim=1)
    return spectro

def diff_to_lr(a2, b2, a1, b1):
    left_real = (a1 + a2) / 2
    left_imag = (b1 + b2) / 2
    right_real = (a1 - a2) / 2
    right_imag = (b1 - b2) / 2
    left_spectro = torch.cat((left_real.unsqueeze(1), left_imag.unsqueeze(1)), dim=1)
    right_spectro = torch.cat((right_real.unsqueeze(1), right_imag.unsqueeze(1)), dim=1)
    return left_spectro, right_spectro

class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split('.')[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",                                     # overwrite existing file
            "-f", "rawvideo",                         # file format
            "-s", "{}x{}".format(shape[1], shape[0]), # size of one frame
            "-pix_fmt", "rgb24",                      # 3 channels
            "-r", str(self.fps),                      # frames per second
            "-i", "-",                                # input comes from a pipe
            "-an",                                    # not to expect any audio
            "-vcodec", self.vcodec,                   # video codec
            "-pix_fmt", "yuv420p",                  # output video in yuv420p
            self.file]

        self.pipe = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10**9)

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print('Process running overtime! Killed.')


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()

# for fair-play
def extract_video(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-ss' + " "
    command1 += '00:00:2.09' + " "
    command1 += "-c " + "copy " + "-t "
    command1 += '00:00:8.00' + " "
    command1 += '-an' + " " + dst
    # print(command1)
    #    print command1
    os.system(command1)
    return

# for sop_duet
# def extract_video(video, dst):
#     command1 = 'ffmpeg '
#     command1 += '-i ' + video + " "
#     command1 += '-ss' + " "
#     command1 += '00:00:12.0' + " "
#     command1 += "-c " + "copy " + "-t "
#     command1 += '00:00:6.3' + " "
#     command1 += '-an' + " " + dst
#     # print(command1)
#     #    print command1
#     os.system(command1)
#     return

def combine_audio(left, right, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + left + " "
    command1 += '-i ' + right + " "
    command1 += '-filter_complex' + " "
    command1 += 'amerge=inputs=2' + " "
    command1 += dst
    # print(command1)
    #    print command1
    os.system(command1)
    return

def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = ["ffmpeg", "-y",
               "-loglevel", "quiet",
               "-i", src_video,
               "-i", src_audio,
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               dst_video]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.)

        if verbose:
            print('Processed:{}'.format(dst_video))
    except Exception as e:
        print('Error:[{}] {}'.format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, 'video should be in 4D numpy array'
    L, H, W, C = tensor.shape
    writer = VideoWriter(
        path,
        fps=fps,
        shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)
