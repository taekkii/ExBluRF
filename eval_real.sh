# run script
# usage
# bash run.sh gpu# expname scene hold expname_extra

gpu=$1
expname=$2
scene_name=$3
reso="[[128,128,128],[256,256,128],[512,512,128],[1408,1156,128]]"

expname_full="${2}_${4}"

#for scene_name in "blurball" "blurbasket" "blurbuick" "blurcoffee" "blurdecoration" "blurgirl" "blurheron" "blurparterre" "blurpuppet" "blurstair"; do
CUDA_VISIBLE_DEVICES=$gpu python run_nerf.py \
                            --config configs/${expname}/${scene_name}.txt \
                            --reso ${reso} \
                            --expname ${scene_name} \
                            --basedir ./results/${expname_full}/real_motion_blur \
                            --tbdir ./results/${expname_full}/real_motion_blur \
                            --export_colmap

CUDA_VISIBLE_DEVICES=$gpu python run_nerf.py \
                            --config configs/${expname}/${scene_name}.txt \
                            --reso ${reso} \
                            --expname ${scene_name} \
                            --basedir ./results/${expname_full}/real_motion_blur \
                            --tbdir ./results/${expname_full}/real_motion_blur \
                            --eval_only

#done