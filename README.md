# LAVSS
Pytorch implementation for "LAVSS: Location-Guided Audio-Visual Spatial Audio Separation".
<div align=center><img src="https://github.com/YYX666660/YYX666660.github.io/blob/main/LAVSS/teaser_figure.png" width="690" height="230" /></div>

Unlike most of previous monaural audio-visual separation (MAVS) works, we put forward ***audio-visual spatial audio separation***. we explicitly exploit positional representations of sounding objects as an additional spatial guidance for separating individual binaural sounds. With a LAVSS framework, our model can simultaneously address location-guided object audio-visual separation tasks.  
  
## Environment
The code is developed under the following configurations.
* Hardware: 1-4 GPUs (change ``os.environ["CUDA_VISIBLE_DEVICES"]`` accordingly)
* Software: Ubuntu 16.04.3 LTS, ***CUDA>=8.0, Python>=3.5, PyTorch>=1.2***

## Training
The frames and detection results used in the paper can be downloaded from this link. If you want to use a new dataset, please follow the following steps.  
  
  
1. Prepare video dataset.  

    a. Download FAIR-Play dataset from:  
  https://github.com/facebookresearch/FAIR-Play  
  
    b. Download videos.


2. Preprocess videos. You can do it in your own way as long as the index files are similar.

    a. Extract frames at 8fps and waveforms at 11025Hz from videos. We have following directory structure (please first ignore the detection results):

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

    b. We created the index files ``train.csv/val.csv/test.csv``. And the files are with the following format:
    
    ```
    ./data/binaural_audios/000541.wav,./data/frames/000541.mp4,82
    ./data/binaural_audios/000137.wav,./data/frames/000137.mp4,82
    ```
    
    For each row, it stores the information: AUDIO_PATH,FRAMES_PATH,NUMBER_FRAMES

    c. Detect objects in video frames. We used object detector trained by Ruohan used in his Cosep project (see [CoSep repo](URL "https://github.com/rhgao/co-separation")). The detected objects for each video are stored in a .npy file.

