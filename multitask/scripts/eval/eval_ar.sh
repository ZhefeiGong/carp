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

# the default order of the tasks as following
tasks=("coffee" "hammer" "mug" "nut" "square" "stack" "stackthree" "thread")
task_ids=(0 1 2 3 4 5 6 7)

# set the datasets to evaluate
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


############################################################################################################################################
for i in "${!tasks[@]}"; do
  task=${tasks[$i]}
  dataset_path=${dataset_paths[$i]}
  task_id=${task_ids[$i]}
  
  python eval_ar.py \
      --output_dir "/path/to/your/output/dir/**ep_${task}" \
      --ar_ckpt "/path/to/ar/ckpt/ar-ep_**.pth" \
      --dataset_path "$dataset_path" \
      --nactions 8 \
      --max_steps 400 \
      --task_id "$task_id"
done



