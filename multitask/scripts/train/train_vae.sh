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

# the name of each dimension
declare -a dims=("x" "y" "z" "r1" "r2" "r3" "r4" "r5" "r6" "gripper")

# the default order of the tasks : `coffee`, `hammer`, `mug`, `nut`, `square`, `stack`, `stack_three`, `threading`
data_paths=(
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
export OMP_NUM_THREADS=16
dim_index=0
for dim in "${dims[@]}"; do
    model_name="coffee_hammer_mug_nut_square_stack_stackthree_thread-cos-${dim}"
    echo "Running training for ${model_name}"
    
    torchrun --nproc_per_node=1 --nnodes=1 train_vae.py \
        --bs 500 \
        --ep 50 \
        --model_name "$model_name" \
        --exp_name 'vq' \
        --seed 42 \
        --vocab_size 1024 \
        --act_dim_sep $dim_index \
        --vqnorm True \
        --saving_interval 10 \
        --workers 16 \
        --data_paths "${data_paths[@]}"
    
    ((dim_index++))
done


