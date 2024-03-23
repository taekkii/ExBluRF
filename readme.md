
# ExBluRF
This is official implementation of [ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_ExBluRF_Efficient_Radiance_Fields_for_Extreme_Motion_Blurred_Images_ICCV_2023_paper.pdf) (ICCV 2023).

## Quickstart

### Local Setup

```shell
git clone https://github.com/taekkii/ExBluRF.git
cd ExBluRF
conda env create --file environment.yml
conda activate exblurf
```

### Exblur Dataset

You can download Exblur dataset from [here](https://drive.google.com/drive/folders/1zTLW9kPe8lVgl8U2RkSHI4Tm5HyuCAon?usp=sharing)
Extreme synthetic can be also found from above link.

Download and place it under this repo in below format

```
./data/exblur
./data/synthetic
```
Note that "train.txt" is unused component of data-loading.

### DeblurNeRF Dataset

You can download Real Camera Motion Blur dataset from DeblurNeRF authors' drive: [here](https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx).
We thank for [DeblurNeRF](https://github.com/limacv/Deblur-NeRF) authors for providing the dataset.

Download and place the directory under this repo in below format
```
./data/real_camera_motion_blur
```

### Train & Evaluate

```shell
bash eval_exblur.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # exblur
bash eval_synthetic.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # synthetic
bash eval_real.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # real-motion-blur
```

### Train Only

```shell
bash run_exblur.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # exblur
bash run_synthetic.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # synthetic
bash run_real.sh ${GPU} ${EXPNAME} ${SCENE_NAME} # real-motion-blur
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
This source code is derived from the implementation of [DeblurNeRF](https://github.com/limacv/Deblur-NeRF) and [Plenoxels](https://github.com/sxyu/svox2).
We appreciate the effort of the contributor to that repository.


