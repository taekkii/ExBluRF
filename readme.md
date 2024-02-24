
# ExBluRF
This is official implementation of [ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_ExBluRF_Efficient_Radiance_Fields_for_Extreme_Motion_Blurred_Images_ICCV_2023_paper.pdf) (ICCV 2023).

### Quickstart

#### Local Setup

```shell
git clone https://github.com/taekkii/ExBluRF.git
cd ExBluRF
conda env create --file environment.yml
conda activate exblurf
```

#### Exblur Dataset

You can download Exblur dataset from [here](https://drive.google.com/drive/folders/1zTLW9kPe8lVgl8U2RkSHI4Tm5HyuCAon?usp=sharing)
Extreme synthetic can be also found from above link.

Download and place it under this repo in below format

```
./data/exblur
./data/synthetic
```

### DeblurNeRF Dataset

You can download real_camera_motion_blur dataset [here](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx).
We thank for [DeblurNeRF](https://github.com/limacv/Deblur-NeRF) authors for providing the dataset.

Download and place the directory under this repo in below format
```
./data/real_camera_motion_blur
```

### train & evaluate

```shell
bash eval_exblur.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # real-motion-blur
bash eval_synthetic.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # exblur
bash eval_real.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # synthetic
```

### train

```shell
bash run_exblur.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # real-motion-blur
bash run_synthetic.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # exblur
bash run_real.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # synthetic
```


## Citation
If you find this useful, please consider citing our paper:
```
@inproceedings{lee2023exblurf,
    Title        = {ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images},
    Author       = {Lee, Dongwoo and Oh, Jeongtaek and Rim, Jaesung and Cho, Sunghyun and Lee, Kyoung Mu},
    Booktile     = {ICCV},
    Year         = {2023}
}
```


## Acknowledge
This source code is derived from the famous pytorch reimplementation of [DeblurNeRF](https://github.com/limacv/Deblur-NeRF) and [Plenoxels](https://github.com/sxyu/svox2).
We appreciate the effort of the contributor to that repository.


