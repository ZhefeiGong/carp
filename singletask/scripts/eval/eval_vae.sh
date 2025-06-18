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

# set the datasets to evaluate (currently, only support robomimic tasks)
dataset_path="/path/to/dataset/***.hdf5"

# set the base folder path
vae_ckpt_base_paths=(
  "/path/to/x-dim/vae/folder"
  "/path/to/y-dim/vae/folder"
  "/path/to/z-dim/vae/folder"
  "/path/to/r1-dim/vae/folder"
  "/path/to/r2-dim/vae/folder
  "/path/to/r3-dim/vae/folder"
  "/path/to/r4-dim/vae/folder
  "/path/to/r5-dim/vae/folder"
  "/path/to/r6-dim/vae/folder
  "/path/to/gripper-dim/vae/folder
)
save_path="/path/to/your/save/folder"

############################################################################################################################################
# each epoch (choose according to your need)
for step in 50 100 ...; do
  vae_ckpt_paths=()
  for base_path in "${vae_ckpt_base_paths[@]}"; do
    vae_ckpt_paths+=("/path/to/your/train/folder/${base_path}/vae-ckpt-${step}.pth")
  done

  echo "Evaluating ckpt step: ${step}"

  python eval_vae.py \
    --is_vis_detail \
    --save_path "${save_path}/step${step}" \
    --dataset_path "$dataset_path" \
    --vae_ckpt_paths "${vae_ckpt_paths[@]}"
done


