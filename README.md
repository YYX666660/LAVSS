# LAVSS
Pytorch implementation for "LAVSS: Location-Guided Audio-Visual Spatial Audio Separation".
<div align=center><img src="https://github.com/YYX666660/YYX666660.github.io/blob/main/LAVSS/teaser_figure.png" width="690" height="230" /></div>

Unlike most of previous monaural audio-visual separation (MAVS) works, we put forward ***audio-visual spatial audio separation***. we explicitly exploit positional representations of sounding objects as an additional spatial guidance for separating individual binaural sounds. With a LAVSS framework, our model can simultaneously address location-guided object audio-visual separation tasks.  
  
# Environment
The code is developed under the following configurations.
* Hardware: 1-4 GPUs (change ``os.environ["CUDA_VISIBLE_DEVICES"]`` accordingly)
* Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=1.2***

# Training
The frames and detection results used in the paper can be downloaded from this link. If you want to use a new dataset, please follow the following steps.

1. Extract frames at 8fps and waveforms at 11025Hz from videos. We have following directory structure (please first ignore the detection results):
```
data
├── binaural_audios
|   ├── 000001.wav
│   ├── 000002.wav
│   |...
|
└── frames
|   ├── 000001.mp4
│   |   ├── 000001.jpg
│   |   ├── 000002.jpg
│   |   ├── ...
│   ├── 000002.mp4
│   |   ├── 000001.jpg
│   |   ├── 000002.jpg
│   |   ├── ...
│   ├── ...
|
└── detection_results
|   ├── 000001.mp4.npy
│   ├── 000002.mp4.npy
│   ├── ...
```
