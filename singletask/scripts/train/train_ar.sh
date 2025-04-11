#!/bin/bash

# apt install
apt update
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# env activate
source /activate/your/miniforge3/here
conda activate carp-st

# folder
cd /path/to/singletask

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

############################################################################################################################################
# run - state
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_ar.py \
--bs=256 \
--ep=4000 \
--model_name='*task*-state-ly8-b256g1-o2-em64' \
--exp_name='ar' \
--tdepth=8 \
--tembed=64 \
--tnobs=2 \
--seed=42 \
--vocab_size=512 \
--workers=16 \
--saving_interval=50 \
--topk=10 \
--is_rollout_during_train=True \
--data_path="/path/to/state-based/dataset/***.hdf5" \
--vae_ckpt_paths \
"/path/to/vae/ckpt/x-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/y-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/z-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r1-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r2-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r3-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r4-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r5-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r6-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/gripper-vae-ckpt-***.pth" \
--act_dim_names 'x' 'y' 'z' 'r1' 'r2' 'r3' 'r4' 'r5' 'r6' 'gripper' \


# run - image
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=1 --nnodes=1 train_ar.py \
--bs=64 \
--ep=4000 \
--model_name='*task*-image-ly16-b64g1-o1-em160' \
--exp_name='ar' \
--tdepth=16 \
--tembed=160 \
--tnobs=1 \
--seed=42 \
--vocab_size=512 \
--workers=16 \
--saving_interval=50 \
--topk=10 \
--is_rollout_during_train=True \
--data_path="/path/to/image-based/dataset/***.hdf5" \
--vae_ckpt_paths \
"/path/to/vae/ckpt/x-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/y-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/z-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r1-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r2-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r3-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r4-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r5-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/r6-vae-ckpt-***.pth" \
"/path/to/vae/ckpt/gripper-vae-ckpt-***.pth" \
--act_dim_names 'x' 'y' 'z' 'r1' 'r2' 'r3' 'r4' 'r5' 'r6' 'gripper' \


# e.g. for kitchen state
# export LD_LIBRARY_PATH=/path/to/envs/carp-st/lib/python3.9/site-packages/mujoco # probably need this
# --data_path="/path/to/dataset/kitchen/kitchen_demos_multitask"
# --act_dim_names 'x' 'y' 'z' 'r1' 'r2' 'r3' 'r4' 'g1' 'g2'


