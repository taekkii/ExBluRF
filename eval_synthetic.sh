# run script
# usage
# bash run.sh gpu# expname scene hold expname_extra

gpu=$1
expname=$2
scene_name=$3
reso="[[128,128,128],[256,256,256],[512,512,256]]"

expname_full="${2}_${4}" 

CUDA_VISIBLE_DEVICES=$gpu python run_nerf.py \
                            --config configs/${expname}/${scene_name}.txt \
                            --reso ${reso} \
                            --expname ${scene_name} \
                            --basedir ./results/${expname_full}/synthetic \
                            --tbdir ./results/${expname_full}/synthetic \
                            --eval_only

