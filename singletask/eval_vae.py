import os
os.environ["MUJOCO_GL"]="osmesa"
import torch
from copy import deepcopy
import collections
from tqdm import tqdm
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import matplotlib.pyplot as plt
import pathlib
import wandb
import wandb.sdk.data_types.video as wv
import dill
import json
import h5py
import numpy as np
from typing import Dict, List
from utils.train_util import load_shape_meta
from omegaconf.omegaconf import open_dict
from env.common.pytorch_util import dict_apply
from env.common.normalizer import LinearNormalizer
from env.gym_util.async_vector_env import AsyncVectorEnv
from env.gym_util.sync_vector_env import SyncVectorEnv
from env.gym_util.multistep_wrapper import MultiStepWrapper
from env.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from env.common.rotation_transformer import RotationTransformer
from env.common.pytorch_util import dict_apply
from env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from env.common.rotation_transformer import RotationTransformer
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from svqvae import build_vae_disc

rotation_transformer = RotationTransformer(from_rep='axis_angle', 
                                            to_rep='rotation_6d', # euler_angles | rotation_6d
                                            from_convention=None,
                                            to_convention=None)

def build(device, vae_ckpt):
    """
    @func: build the model and data normalizer
    """
    # load args
    args = torch.load(vae_ckpt, map_location='cpu')['args']
    print(args)
    # build vqvae
    vae = build_vae_disc(
        device=device,
        # encoder | decoder
        V=args['vocab_size'], 
        Cvae=args['vocab_ch'],
        ch=args['vch'],
        ch_mult= (2, 4), # or args['vch_mult_ls'],
        action_dim=1, # NOTE: vqvae for each separate action
        num_actions=args['act_horizon'],
        dropout=args['vdrop'],
        # quant
        beta=args['vqbeta'],
        using_znorm=args['vqnorm'],
        quant_conv_ks=3, # fixed here
        quant_resi=args['vqresi'],
        share_quant_resi=len(args['patch_nums']),
        patch_nums=args['patch_nums'],
        vae_init=args['vae_init'],
        vocab_init=args['vocab_init'],
    )
    # load checkpoint
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_wo_ddp'], using_znorm = args['vqnorm']) # cosine | euler
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    print(f'[INFO] vae finished')
    # load norm
    normalizer = LinearNormalizer()
    norm_local=torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_norm']
    normalizer.load_state_dict(norm_local)
    normalizer.to(device)
    print(f'[INFO] normalizer finished')
    return normalizer, vae
    
def choose_config_data(dataset_path, traj_indices):
    """
    @func: 
    choose the config data for test
    """
    data=list()
    with h5py.File(dataset_path) as file:
        demos = file['data']
        for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
            if i in traj_indices:
                demos = file['data']
                demo = demos[f'demo_{i}']
                ## Actions
                actions = demo['actions'][:].astype(np.float32)
                pos = actions[...,:3]
                rot = actions[...,3:6]
                gripper = actions[...,6:]
                rot = rotation_transformer.forward(rot)
                actions = np.concatenate([
                    pos, rot, gripper
                ], axis=-1).astype(np.float32)
                ## States
                states = demo['states'][:].astype(np.float32)
                data.append({'actions': actions, 'states': states})
    return data

def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env

def build_env(dataset_path, 
              traj_indices,
              save_path,
              env_n_obs_steps = 1,
              env_n_action_steps = 16,
              max_steps=700,
              is_vis_detail=False,
              obs_keys=None):
    """
    @func: build the simulation environments
    """
    # set environment
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['controller_configs']['control_delta'] = False # absolute position control rather than delta/relative/velocity control
    crf=22
    fps=10
    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)
    n_envs=len(traj_indices)
    def env_fn():
        robomimic_env = create_env(
            env_meta=env_meta, 
            obs_keys=obs_keys
        )
        robomimic_env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                RobomimicLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=obs_keys,
                    init_state=None,
                    render_hw=(128,128),
                    render_camera_name='agentview',
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=env_n_obs_steps,
            n_action_steps=env_n_action_steps,
            max_episode_steps=max_steps
        )
    env_fns = [env_fn] * n_envs * 2 # one for raw and one for vqvae
    env_seeds = list()
    env_prefixs = list()
    env_init_fn_dills = list()
    # raw env
    with h5py.File(dataset_path, 'r') as f:
        for idx, d in enumerate(data):
            init_state = d['states'][0]
            def init_fn(env, 
                        init_state=init_state,
                        enable_render=is_vis_detail):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(save_path).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                # switch to init_state reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = init_state
            env_seeds.append(traj_indices[idx])
            env_prefixs.append('RAW')
            env_init_fn_dills.append(dill.dumps(init_fn)) # to byte stream
    # vqvae env
    with h5py.File(dataset_path, 'r') as f:
        for idx, d in enumerate(data):
            init_state = d['states'][0]
            def init_fn(env, 
                        init_state=init_state,
                        enable_render=is_vis_detail):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(save_path).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                # switch to init_state reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = init_state
            env_seeds.append(traj_indices[idx])
            env_prefixs.append('VQVAE')
            env_init_fn_dills.append(dill.dumps(init_fn)) # to byte stream
    # get env
    env = AsyncVectorEnv(env_fns)
    n_envs = len(env_fns)
    n_inits = len(env_init_fn_dills)
    # return
    return env, n_inits, env_init_fn_dills, env_meta, env_seeds, env_prefixs

def run_env(env, 
            n_inits, 
            env_init_fn_dills, 
            env_meta, 
            env_seeds, 
            env_prefixs, 
            data,
            normalizers,
            vaes,
            save_path,
            slice_size=16,
            max_steps=700,
            is_vis_detail=False):
    """
    @func: 
    run the env for test
    """
    ### begin to run
    # allocate data
    all_video_paths = [None] * n_inits
    all_rewards = [None] * n_inits
    env.call_each('run_dill_function', args_list=[(x,) for x in env_init_fn_dills])
    obs = env.reset()
    env_name = env_meta['env_name']
    pbar = tqdm(total=max_steps, desc=f"Eval {env_name}Image", leave=False)
    # preprocess the actions
    import math
    actual_max_length = max(len(d['actions']) for d in data)
    margin_max_length = math.ceil(actual_max_length / slice_size) * slice_size
    padded_actions = []
    for d in data:
        padded_act = np.vstack((d['actions'], np.tile(d['actions'][-1], (margin_max_length - len(d['actions']), 1))))
        padded_actions.append(padded_act)
    padded_actions = np.array(padded_actions) # BLC
    B,L,C = padded_actions.shape # BLC
    actions_run = np.split(padded_actions, L // slice_size, axis=1) # BLC
    # run
    done = False
    act_idx=0
    loss = 0
    act_repo_raw=[]
    act_repo_vae=[]
    act_dim = 10 # fix the action dimension for robomimic tasks
    while not done:
        ## check exit
        if act_idx >= len(actions_run):
            break
        ## reconstruct
        with torch.no_grad():
            ## raw
            action_raw = actions_run[act_idx] # [B,num_actions,action_dim] ｜  [B,16,7]
            ## vae
            action_vae = deepcopy(action_raw) # [B,num_actions,action_dim] ｜  [B,16,7]
            action_vae = normalizers[0]['action'].normalize(action_vae) # BL7
            action_vae = action_vae.view(B,1,slice_size,C).contiguous() # [B,1,num_actions,action_dim] | [B,1,16,10]
            rec_action_vae = []
            for idx in range(act_dim): # calculate at each action dimension
                rec_action_vae.append(vaes[idx].inp_to_action(action_vae[:,:,:,idx:idx+1])) # [B,1,16,1]
            rec_action_vae = torch.cat(rec_action_vae, axis=-1)
            rec_action_vae = normalizers[0]['action'].unnormalize(rec_action_vae) # [B,1,num_actions,action_dim] | [B,1,16,10]
            rec_action_vae = rec_action_vae.to('cpu').numpy().squeeze(axis=1) # [B,num_actions,action_dim] | [B,16,10]
            ## execute
            loss += np.mean((action_raw - rec_action_vae) ** 2) # [B,16,7] * [B,16,7]
            env_action = np.concatenate((action_raw, rec_action_vae), axis=0) # [2*B,16,7]
            ## transform 
            pos = env_action[...,:3]
            rot = env_action[...,3:-1]
            gripper = env_action[...,-1:]
            rot = rotation_transformer.inverse(rot)
            env_action = np.concatenate([
                pos, rot, gripper
            ], axis=-1).astype(np.float32)
            ## sign
            act_idx+=1
        ## storage
        act_repo_raw.append(action_raw)
        act_repo_vae.append(rec_action_vae)
        ## run
        obs, reward, done, info = env.step(env_action)
        done = np.all(done)
        pbar.update(env_action.shape[1])
    pbar.close()
    # collect data for this round
    loss = loss/(len(data[0]['actions'])//slice_size)
    np.save(f"{save_path}/act_raw.npy", np.concatenate(act_repo_raw, axis=1))
    np.save(f"{save_path}/act_vae.npy", np.concatenate(act_repo_vae, axis=1))
    print(f'[INFO] mse loss : {loss}')
    # log
    if is_vis_detail:
        all_video_paths = env.render()
        all_rewards = env.call('get_attr', 'reward')
        log_data = dict()
        for i in range(n_inits):
            seed = env_seeds[i]
            prefix = env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            log_data[f'{prefix}_sim_max_reward_{seed}'] = max_reward
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[f'{prefix}_sim_video_{seed}'] = sim_video
        json_log = dict()
        for key, value in log_data.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        json_log['mse_loss'] = loss
        out_path = os.path.join(save_path, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    return

def vis_act_raw2vae_linechart(dimensions, act_dim, raw_pth, vae_pth, save_pth, B_idx, save_png_name, is_rotation_6d=True):    
    matrix_raw = np.load(raw_pth)
    matrix_vae = np.load(vae_pth)
    print('the martix has shape : ', matrix_raw.shape )
    assert matrix_raw.shape == matrix_vae.shape , "Shape mismatch between raw and VAE matrices"
    fig, axs = plt.subplots(act_dim, 1, figsize=(10, 20))
    x_ticks = range(0, matrix_raw.shape[1], 16)
    loss = np.mean((matrix_raw[B_idx, :, :] - matrix_vae[B_idx, :, :])**2)
    for i in range(act_dim):
        axs[i].plot(matrix_raw[B_idx, :, i], label='raw', color='blue')
        axs[i].plot(matrix_vae[B_idx, :, i], label='vae', color='red')
        axs[i].legend()
        axs[i].set_title(dimensions[i])
        axs[i].set_xlabel('Action')
        axs[i].set_ylabel('Value')
        for x_tick in x_ticks:
            axs[i].axvline(x=x_tick, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    loss_str = "{:.5f}".format(loss)
    img_path = f"{save_pth}/{save_png_name}_{loss_str}.png"
    plt.savefig(img_path, dpi=300)
    print(f"Image saved to {img_path}")

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--vae_ckpt_paths", type=str, required=False, nargs="+", help="path/to/the/vae/checkpoints") # NOTE: need
    parser.add_argument("--dataset_path", type=str, required=True, help="path/to/the/dataset") # NOTE: need | currently, only Robomimic tasks are supported for VQ-VAE evaluation
    parser.add_argument("--save_path", type=str, required=True, help="path/to/save/the/results") # NOTE: need
    parser.add_argument("--traj_indices", type=int, nargs='+', default=[23,48,51,70,93,122,156,160,181,192], help="Trajectory indices") # NOTE: you can choose randomly here within the range of the dataset
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum number of steps")
    parser.add_argument("--env_n_obs_steps", type=int, default=2, help="Number of observation steps")
    parser.add_argument("--env_n_action_steps", type=int, default=16, help="Number of action steps")
    parser.add_argument("--slice_size", type=int, default=16, help="Slice size")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--is_vis_detail", action='store_true', help="Whether visualize the evaluation results here")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    ### params
    args = get_args()
    device = 'cuda:0'
    os.makedirs(args.save_path, exist_ok=True)
    
    ### get models
    normalizers=[]
    vaes=[]
    for ckpt in args.vae_ckpt_paths:
        normalizer, vae = build(device, ckpt)
        normalizers.append(normalizer)
        vaes.append(vae)
    
    ### get data
    data = choose_config_data(args.dataset_path, 
                              args.traj_indices )
    obs_keys=['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
    
    ### build sim envs
    env, n_inits, env_init_fn_dills, env_meta, env_seeds, env_prefixs = build_env(args.dataset_path, 
                                                                                  args.traj_indices, 
                                                                                  args.save_path, 
                                                                                  args.env_n_obs_steps, 
                                                                                  args.env_n_action_steps, 
                                                                                  args.max_steps,
                                                                                  args.is_vis_detail,
                                                                                  obs_keys)
    
    ### run the envs
    run_env(env, 
            n_inits, 
            env_init_fn_dills, 
            env_meta, 
            env_seeds, 
            env_prefixs, 
            data, 
            normalizers, 
            vaes, 
            args.save_path, 
            args.slice_size, 
            args.max_steps,
            args.is_vis_detail)
    
    ### visualization
    root_pth = args.save_path
    save_pth = root_pth + '/images'
    raw_pth = root_pth + f'/act_raw.npy'
    vae_pth = root_pth + '/act_vae.npy'
    tmp_args = torch.load(args.vae_ckpt_paths[0], map_location='cpu')['args']
    dimensions = tmp_args['act_dim_names']
    act_dim = tmp_args['act_dim']
    os.makedirs(save_pth, exist_ok=True) 
    for B_idx in range(len(args.traj_indices)):
        traj_idx = args.traj_indices[B_idx]
        save_png_name = f"act_{traj_idx}"
        vis_act_raw2vae_linechart(dimensions, act_dim, raw_pth, vae_pth, save_pth, B_idx, save_png_name)
    
    
    