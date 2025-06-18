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
python eval_ar.py \
--output_dir "/path/to/your/output/folder" \
--ar_ckpt "/path/to/your/ckpt/***.pth" \
--dataset_path "/path/to/corresponding/dataset/***" \
--nactions 8 \
--max_steps 400 \
--vis_rate 1.0 \


# e.g. for pusht-state
# --data_path="/path/to/dataset/pusht/pusht_cchi_v7_replay.zarr"
# --max_steps 300
# --is_mm_vis # enable visualization of trajectories from a fixed initialization to highlight multi-modal predictions.
# --sample_top_k 5 # number of top tokens to sample from during inference; a larger value increases multi-modal diversity; setting to 1 yields greedy decoding (highest probability token).

# e.g. for kitchen-state
# export LD_LIBRARY_PATH=/path/to/envs/carp-st/lib/python3.9/site-packages/mujoco # probably need this
# --data_path="/path/to/dataset/kitchen/kitchen_demos_multitask"
# --max_steps 280

# e.g. for robomimic-state
# --data_path="/path/to/dataset/robomimic/datasets/***/ph/low_dim_abs.hdf5"
# --max_steps 400

# e.g. for robomimic-image
# --data_path="/path/to/dataset/robomimic/datasets/***/ph/image_abs.hdf5"
# --max_steps 400


