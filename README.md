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
    
    For each row, it stores the information: ``AUDIO_PATH,FRAMES_PATH,NUMBER_FRAMES``

    c. Detect objects in video frames. We used object detector trained by Ruohan used in his Cosep project (see [CoSep repo](URL "https://github.com/rhgao/co-separation")). The detected objects for each video are stored in a .npy file.

3. Train the non-positional model for warming up. The trained MUSIC model paths used in the paper can be downloaded from this link.
```
./scripts/train_mono.sh
```
4. Train the position-guided audio-visual separation model.
```
./scripts/train.sh
```
5. During training, visualizations are saved in HTML format under data/ckpt/MODEL_ID/visualization/.

## Evaluation
1. Evaluate the trained LAVSS model. Our pre-trained model can be downloaded from here. Please put it into data/ckpt.
```
./scripts/eval.sh
```

## Acknowledgement
We borrowed a lot of code from [SoP](URL "https://github.com/hangzhaomit/Sound-of-Pixels") and [CCoL](URL "https://github.com/YapengTian/CCOL-CVPR21"), and used detector from Ruohan' [CoSep](URL "https://github.com/rhgao/co-separation"). We thank the authors for sharing their code. If you use our codes, please also cite their nice works.

## Citation
```
@inproceedings{ye2024lavss,
  title={Lavss: Location-guided audio-visual spatial audio separation},
  author={Ye, Yuxin and Yang, Wenming and Tian, Yapeng},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5508--5519},
  year={2024}
}
```
