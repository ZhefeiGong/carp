# CARP | Single-Task

## 🎏 Introduction
We evaluate our method on the `lift`, `can`, and `square` tasks from [Robomimic](https://robomimic.github.io/), as well as the Franka `kitchen` task from [Relay Policy Learning](https://arxiv.org/pdf/1910.11956), following the same experimental setup as [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

## 🛠️ Setup

* Build [miniforge](https://github.com/conda-forge/miniforge#mambaforge) virtual environment.
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
* Build `carp-st` environment.
```bash
conda env create -f environment.yaml
```
* For `mujoco` on `Ubuntu 20.04`, we need to install the following apt packages:
```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

## 📊 Dataset

* Download the Robomimic datasets in both [state-based](https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip) and [image-based](https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip) formats. The Kitchen dataset can be downloaded [here](https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip).
* For Robomimic, we use the `absolute-action` setting (dataset files ending with `abs.hdf5`). If you are using `relative-action` datasets instead, please refer to `singletask/env/dataset/robomimic_dataset_conversion.py` for conversion reference.

## 🚄 Training

### Multi-Scale Action Tokenization (MSAT)
As described in the paper, CARP consists of two stages. The first step is training a multi-scale action tokenizer to obtain representations of the action sequence at different scales.
To train the tokenizers, run:
```bash
bash ./scripts/train/train_vae.sh
```

### Coarse-to-Fine Autoregressive Prediction (CFAP)
With the action tokenizers, you can train the coarse-to-fine autoregressive model by running:
```bash
bash ./scripts/train/train_ar.sh
```

## 🤖 Evaluation

* To evaluate CARP's performance, run the following command. 
```bash
bash ./scripts/eval/eval_ar.sh
```

* Additionally, we provide a basic check of action tokenization, through visulizing the differences between the reconstructed and raw actions.
```bash
bash ./scripts/eval/eval_vae.sh
```


## 📃 File Structure

```
carp/singletask
├── CFAP  # Coarse-to-Fine Autoregressive model  
│   ├── __init__.py  # Model initialization  
│   ├── autoreg.py  # Autoregressive model  
│   └── basic_ar.py  # Basic implementation  
├── env  # Robomimic and Kitchen environment dependencies  
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
* If you encounter any issues during installation, feel free to open an issue or reach out for help.



