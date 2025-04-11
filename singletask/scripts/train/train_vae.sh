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

# the name of each dimension
declare -a dims=("x" "y" "z" "r1" "r2" "r3" "r4" "r5" "r6" "gripper") # e.g. for robomimic
# declare -a dims=("x" "y" "z" "r1" "r2" "r3" "r4" "g1" "g2") # e.g. for kitchen

# dataset path
data_path="/path/to/dataset/low_dim_abs.hdf5" # e.g. for robomimic
# data_path="/path/to/dataset/kitchen/kitchen_demos_multitask" # e.g. for kitchen

############################################################################################################################################
export OMP_NUM_THREADS=16
dim_index=0
for dim in "${dims[@]}"; do
    model_name="***-cos-${dim}"
    echo "Running training for ${model_name}"
    
    torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
        --bs 256 \
        --ep 400 \
        --model_name "$model_name" \
        --exp_name 'vq' \
        --seed 42 \
        --vocab_size 512 \
        --act_dim_sep $dim_index \
        --vqnorm True \
        --saving_interval 50 \
        --act_dim_names "${dims[@]}" \
        --workers 16 \
        --data_path $data_path
    
    ((dim_index++))
done
