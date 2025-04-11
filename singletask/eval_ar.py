import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import os
os.environ["MUJOCO_GL"]="osmesa" # NOTE: Rendering on CPU (noticeable differences compared to GPU rendering)
import torch
import click
import random
import numpy as np
from env.runner.robomimic_image_runner import RobomimicImageRunner
from env.common.normalizer import LinearNormalizer
from CFAP import build_vae_ar
from utils.train_util import load_shape_meta,load_obs_encoder,save_runner_json
from utils.train_util import load_kitchen_lowdim_runner, load_robomimic_lowdim_runner, load_robomimic_image_runner

@click.command()
@click.option('--output_dir', default='/path/to/your/output/dir', help='Output directory for results.')
@click.option('--ar_ckpt', default='/path/to/your/ar/model', help='Path to the checkpoint file.')
@click.option('--dataset_path', default='/path/to/the/dataset', help='Path to the dataset file.')
@click.option('--nactions', default=8, type=int, help='Number of actions to execute.')
@click.option('--max_steps', default=400, type=int, help='Maximum steps to perform.')
@click.option('--vis_rate', default=0.0, type=float, help='visualization rate.')
def main(output_dir, ar_ckpt, dataset_path, nactions, max_steps, vis_rate):
    
    # build a folder
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d%H")
    output_dir = f"{output_dir}_{current_time}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Directory '{output_dir}' created.")
    else:
        print(f"[INFO] Directory '{output_dir}' already exists.")
    
    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] device : {device}')
    
    ### load obs encoder
    if 'image' in dataset_path:
        shape_meta=load_shape_meta()
        obs_encoder = load_obs_encoder(shape_meta)
    else:
        obs_encoder = None
    
    ### load args
    args = torch.load(ar_ckpt, map_location='cpu')['args']
    
    ### load vae and ar
    vae, ar = build_vae_ar(
        device=device,
        patch_nums=args['patch_nums'],
        ## multi-scale action tokenization
        V=args['vocab_size'], 
        Cvae=args['vocab_ch'], 
        ch=args['vch'], 
        action_dim=args['act_dim'],
        num_actions=args['act_horizon'],
        dropout=args['vdrop'],
        beta=args['vqbeta'],
        using_znorm=args['vqnorm'],
        quant_conv_ks=3, # fixed
        quant_resi=args['vqresi'],
        share_quant_resi=4, # fixed
        ## coarse-to-fine autoregressive prediction
        obs_encoder = obs_encoder,
        depth=args['tdepth'], 
        n_obs_steps=args['tnobs'],
        obs_dim=args['obs_dim'],
        embed_dim=args['tembed'],
        shared_aln=args['saln'], # whether to use shared adaln
        attn_l2_norm=args['anorm'], # whether to use L2 normalized attention
        init_adaln=args['taln'], # for autoregressive
        init_adaln_gamma=args['talng'], # for autoregressive
        init_head=args['thd'], # for autoregressive
        init_std=args['tini'], # for autoregressive
    )
    
    ### load ar | ema
    ar_wo_ddp=torch.load(ar_ckpt, map_location='cpu')['trainer']['ema_ar_wo_ddp']
    ar.load_state_dict(ar_wo_ddp, strict=True)
    ar.eval()
    for p in ar.parameters(): p.requires_grad_(False)
    
    ### load vae
    vae_local=torch.load(ar_ckpt, map_location='cpu')['trainer']['vae_local']
    vae.load_state_dict(vae_local, strict=True)
    vae.eval()                          
    for p in vae.parameters(): p.requires_grad_(False)     
    
    ### load norm
    normalizer = LinearNormalizer()
    norm_local=torch.load(ar_ckpt, map_location='cpu')['trainer']['ar_norm']
    normalizer.load_state_dict(norm_local)
    
    ###
    ar.to(device)
    ar.eval()
    vae.to(device)
    vae.eval()
    normalizer.to(device)
    normalizer.eval()
    
    del ar_wo_ddp, vae_local, norm_local
    print(f'[INFO] VAE/AR Finished')
    
    ## seed config
    seed=42
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    ## others
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    print(f'[INFO] Policy Initialization Finished')
    
    # kitchen sim env
    num_test=50 # by default
    num_train=6 # by default
    if 'kitchen' in dataset_path:
        env_runner = load_kitchen_lowdim_runner(
            output_dir=output_dir,
            dataset_path=dataset_path,
            max_steps=max_steps,
            n_obs_steps=args['tnobs'],
            n_action_steps=nactions,
            vis_rate=vis_rate,
            num_test=num_test,
            num_train=num_train,
        )
    # robomimic image-based env
    elif 'image' in dataset_path:
        shape_meta=load_shape_meta()
        env_runner = load_robomimic_image_runner(
            output_dir=output_dir,
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            max_steps=max_steps, 
            n_obs_steps=args['tnobs'],
            n_action_steps=nactions,
            vis_rate=vis_rate,
            num_test=num_test,
            num_train=num_train,
        )
    # robomimic state-based env
    else: 
        env_runner = load_robomimic_lowdim_runner(
            output_dir=output_dir,
            dataset_path=dataset_path,
            max_steps=max_steps, 
            n_obs_steps=args['tnobs'],
            n_action_steps=nactions,
            vis_rate=vis_rate,
            num_test=num_test,
            num_train=num_train,
        )
    runner_log = env_runner.run(vae=vae,
                                ar=ar,
                                normalizer=normalizer)
    local_out_json = os.path.join(output_dir, 'eval_log.json')
    save_runner_json(runner_log, local_out_json)
    
if __name__ == '__main__':
    main()


