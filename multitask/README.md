# CARP | Multi-Task

## 🎏 Introduction

We use [MimicGen](https://mimicgen.github.io/) as our multi-task benchmark, selecting the tasks `coffee`, `hammer`, `mug`, `nut`, `square`, `stack`, `stackthree`, and `thread`, following the same setup as [Sparse Diffusion Policy](https://arxiv.org/pdf/2407.01531).

## 🛠️ Setup

* Build [miniforge](https://github.com/conda-forge/miniforge#mambaforge) virtual environment.
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
* Build `carp-mt` environment.
```bash
conda env create -f environment.yaml
```
* For `mujoco` on `Ubuntu 20.04`, we need to install the following apt packages:
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
* For [MimicGen](https://mimicgen.github.io/docs/introduction/installation.html), we need to install the following additional dependencies from source:
```bash
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```
```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install -e .
```
```bash
git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
cd robosuite-task-zoo
git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
pip install -e .
```

## 📊 Dataset

* Download the MimicGen dataset [here](https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core).
* Convert the `relative-action` dataset to an `absolute-action` dataset using the following command:
```bash
bash ./scripts/misc/dataset_transform.sh
```

## 🚄 Get Start

### Multi-Scale Action Tokenization (MSAT)
As described in the paper, CARP consists of two stages. The first step is training a multi-scale action tokenizer to obtain representations of the action sequence at different scales.
To train the tokenizers, run:
```bash
bash ./scripts/train/train_vae.sh
```

### Coarse-to-Fine Autoregressive Prediction (CFAP)
With the action tokenizers (which you can also download [here](https://huggingface.co/zhefeigong/carp/resolve/main/multitask/vae_ckpt.zip?download=true), pre-trained using the above settings), you can train the coarse-to-fine autoregressive model by running:
```bash
bash ./scripts/train/train_ar.sh
```

## 🤖 Evaluation
* To evaluate CARP's performance, run the following command. (For a quick test, we also provide the pre-trained CFAP model [here](https://huggingface.co/zhefeigong/carp/resolve/main/multitask/ar_ckpt.zip?download=true), trained using the above settings.)
```bash
bash ./scripts/eval/eval_ar.sh
```

* Additionally, we provide a basic check of action tokenization, through visulizing the differences between the reconstructed and raw actions.
```bash
bash ./scripts/eval/eval_vae.sh
```


## 📃 File Structure

```
carp/multitask
├── CFAP  # Coarse-to-Fine Autoregressive model  
│   ├── __init__.py  # Model initialization  
│   ├── autoreg.py  # Autoregressive model  
│   └── basic_ar.py  # Basic implementation  
├── env  # MimicGen environment dependencies  
├── MSAT  # Multi-Scale Action Tokenizer  
│   ├── __init__.py  # Model initialization  
│   ├── quant.py  # Multi-scale quantization  
│   └── vqvae.py  # Action tokenizer  
├── optim  # Optimization tools  
│   ├── amp_opt.py  # Mixed precision optimizer  
│   └── lr_control.py  # Adaptive learning rate scheduler  
├── scripts  # Bash scripts for running tasks  
│   ├── eval  
│   │   ├── eval_ar.sh  # Evaluate CFAP  
│   │   └── eval_vae.sh  # Evaluate MSAT  
│   ├── misc  
│   │   └── dataset_transform.sh  # Convert relative actions to absolute  
│   └── train  
│       ├── train_ar.sh  # Train CFAP  
│       └── train_vae.sh  # Train MSAT  
├── svqvae  # Per-dimension multi-scale action tokenizer  
│   ├── __init__.py  # Model initialization  
│   ├── basic_vae.py  # Basic VAE architecture  
│   ├── quant.py  # Multi-scale quantization  
│   └── vqvae.py  # Per-dimension action tokenizer  
├── utils  # Utility functions  
│   ├── arg_util.py  # Argument parsing  
│   ├── data_sampler.py  # Multi-GPU data sampling  
│   ├── helpers.py  # Miscellaneous helper functions  
│   ├── misc.py  # Training logs  
│   ├── robomimic_dataset_conversion.py  # Convert relative to absolute actions  
│   └── train_util.py  # Training utilities  
├── dist.py  # Distributed training  
├── eval_ar.py  # Evaluate CFAP  
├── eval_vae.py  # Evaluate MSAT  
├── train_ar.py  # Train CFAP  
├── train_vae.py  # Train per-dimension action tokenizer  
├── trainer_ar.py  # CFAP trainer  
└── trainer_vae.py  # Action tokenizer trainer  
```


## 😵‍💫 Troubleshooting

* `Exception: Environment Coffee_D0 not found.`: The issue arises from the import mimicgen error, likely caused by version incompatibilities. For more details, please refer to this [discussion](https://github.com/NVlabs/mimicgen/issues/18).




