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

# set the datasets to evaluate
tasks=("coffee" "hammer" "mug" "nut" "square" "stack" "stackthree" "thread")
dataset_paths=(
  "/path/to/dataset/coffee_d0_abs.hdf5"
  "/path/to/dataset/hammer_cleanup_d0_abs.hdf5"
  "/path/to/dataset/mug_cleanup_d0_abs.hdf5"
  "/path/to/dataset/nut_assembly_d0_abs.hdf5"
  "/path/to/dataset/square_d0_abs.hdf5"
  "/path/to/dataset/stack_d0_abs.hdf5"
  "/path/to/dataset/stack_three_d0_abs.hdf5"
  "/path/to/dataset/threading_d0_abs.hdf5"
)
# set the vae ckpts to evaluate
vae_ckpt_paths=(
  "/path/to/vae/ckpt/x-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/y-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/z-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r1-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r2-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r3-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r4-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r5-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/r6-vae-ckpt-**.pth"
  "/path/to/vae/ckpt/gripper-vae-ckpt-**.pth"
)
task_ids=(0 1 2 3 4 5 6 7)


############################################################################################################################################
for i in "${!tasks[@]}"; do
  dataset_path=${dataset_paths[$i]}
  save_path="/path/to/your/output/dir/${tasks[$i]}"
  task_id=${task_ids[$i]}
  
  python eval_vae.py \
      --is_vis_detail \
      --save_path "$save_path" \
      --dataset_path "$dataset_path" \
      --vae_ckpt_paths "${vae_ckpt_paths[@]}"
done




