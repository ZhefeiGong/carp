#!/bin/bash

# apt install
apt update
apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# folder
cd /path/to/carp/multitask

# env
source /activate/your/miniforge3/here

# conda
conda activate carp-mt

python utils/robomimic_dataset_conversion.py \
--input path/to/core/coffee_d0.hdf5 \
--output path/to/core/coffee_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/hammer_cleanup_d0.hdf5 \
--output path/to/core/hammer_cleanup_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/mug_cleanup_d0.hdf5 \
--output path/to/core/mug_cleanup_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/nut_assembly_d0.hdf5 \
--output path/to/core/nut_assembly_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/square_d0.hdf5 \
--output path/to/core/square_d0_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/stack_d0.hdf5 \
--output path/to/core/stack_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/stack_three_d0.hdf5 \
--output path/to/core/stack_three_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

python utils/robomimic_dataset_conversion.py \
--input path/to/core/threading_d0.hdf5 \
--output path/to/core/threading_d0_abs.hdf5 \
--eval_dir path/to/evaluation/output \
--num_workers 32 \

