#!/bin/bash

# apt install
apt update
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# env activate
source /activate/your/miniforge3/here
conda activate carp-mt

# folder
cd /path/to/carp/multitask

# show the number of cpu
lscpu | grep '^CPU(s):'
cat /proc/cpuinfo| grep "cpu cores"| uniq

# NOTICE: 
# the order of the tasks we set as : 
# `coffee`, `hammer`, `mug`, `nut`, `square`, `stack`, `stack_three`, `threading`

# run
export OMP_NUM_THREADS=16
OMP_NUM_THREADS=16 torchrun --nproc_per_node=4 --nnodes=1 train_ar.py \
--bs=512 \
--ep=200 \
--model_name='coffee_hammer_mug_nut_square_stack_stackthree_thread-img-ly32-b512g4-im1-em160' \
--exp_name='ar' \
--tdepth=32 \
--tembed=160 \
--tnobs=1 \
--seed=42 \
--vocab_size=1024 \
--workers=16 \
--saving_interval=4 \
--is_rollout_during_train=True \
--task_num=8 \
--data_paths \
"/path/to/dataset/coffee_d0_abs.hdf5" \
"/path/to/dataset/hammer_cleanup_d0_abs.hdf5" \
"/path/to/dataset/mug_cleanup_d0_abs.hdf5" \
"/path/to/dataset/nut_assembly_d0_abs.hdf5" \
"/path/to/dataset/square_d0_abs.hdf5" \
"/path/to/dataset/stack_d0_abs.hdf5" \
"/path/to/dataset/stack_three_d0_abs.hdf5" \
"/path/to/dataset/threading_d0_abs.hdf5" \
--vae_ckpt_paths \
"/path/to/vae/ckpt/x-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/y-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/z-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r1-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r2-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r3-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r4-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r5-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/r6-vae-ckpt-**.pth" \
"/path/to/vae/ckpt/gripper-vae-ckpt-**.pth" \



