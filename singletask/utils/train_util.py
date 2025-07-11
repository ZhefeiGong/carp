import torch
import wandb
import json
import os
import numpy as np
import torch.nn as nn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
# import robomimic.models.obs_core as obs_core # in multi-task env
import robomimic.models.base_nets as rmbn # in single-task env
import env.model.crop_randomizer as dmvc
from env.common.robomimic_config_util import get_robomimic_config
from env.common.pytorch_util import dict_apply, replace_submodules
from env.runner.robomimic_image_runner import RobomimicImageRunner
from env.runner.robomimic_lowdim_runner import RobomimicLowdimRunner
from env.runner.kitchen_lowdim_runner import KitchenLowdimRunner
from env.dataset.kitchen_mjl_lowdim_dataset import KitchenMjlLowdimDataset
from env.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from env.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from env.dataset.pusht_dataset import PushTLowdimDataset
from env.runner.pusht_keypoints_runner import PushTKeypointsRunner
from env.runner.pusht_keypoints_fixed_ini_runner import PushTKeypointsFixedIniRunner

def load_shape_meta():
    """
    @func: 
    load the shape meta info for image-based task
    """
    shape_meta = {
        "action": {
            "shape": [10]
        },
        "obs": {
            "agentview_image": {
                "shape": [3, 84, 84],
                "type": "rgb"
            },
            "robot0_eef_pos": {
                "shape": [3]
            },
            "robot0_eef_quat": {
                "shape": [4]
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, 84, 84],
                "type": "rgb"
            },
            "robot0_gripper_qpos": {
                "shape": [2]
            }
        }
    }
    return shape_meta

def load_robomimic_image_dataset(data_path, seed, shape_meta, n_obs_steps):
    """
    @func:
    """
    dataset_train = RobomimicReplayImageDataset(abs_action=True, # NOTE: absolute action here
                                                rotation_rep='rotation_6d', # NOTE: rotation_6d to represent rotation space
                                                n_obs_steps=n_obs_steps,
                                                shape_meta=shape_meta,
                                                use_cache=True,
                                                dataset_path=data_path, 
                                                horizon=16,
                                                pad_after=7,
                                                pad_before=1, 
                                                seed=seed,
                                                val_ratio=0.02,)
    dataset_val = dataset_train.get_validation_dataset()
    normalizer = dataset_train.get_normalizer()
    return dataset_train, dataset_val, normalizer

def load_robomimic_lowdim_dataset(data_path, seed, is_only_act=False):
    """
    @func:
    """
    dataset_train = RobomimicReplayLowdimDataset(abs_action=True, # NOTE: absolute action here
                                                 rotation_rep='rotation_6d', # NOTE: rotation_6d to represent rotation space
                                                 dataset_path=data_path,
                                                 horizon=16,
                                                 pad_after=7,
                                                 pad_before=1,
                                                 seed=seed,
                                                 val_ratio=0.02,
                                                 is_only_act=is_only_act)
    dataset_val = dataset_train.get_validation_dataset()
    normalizer = dataset_train.get_normalizer()
    return dataset_train, dataset_val, normalizer

def load_kitchen_lowdim_dataset(data_path, seed, is_only_act=False):
    """
    @func:
    """
    dataset_train = KitchenMjlLowdimDataset(abs_action=True, # NOTE: absolute action here
                                            dataset_dir=data_path,
                                            horizon=16,
                                            pad_after=7,
                                            pad_before=1,
                                            robot_noise_ratio=0.1,
                                            seed=seed,
                                            val_ratio=0.02,
                                            is_only_act=is_only_act)
    dataset_val = dataset_train.get_validation_dataset()
    normalizer = dataset_train.get_normalizer()
    return dataset_train, dataset_val, normalizer

def load_pusht_lowdim_dataset(data_path, seed, is_only_act=False):
    """
    :func:
    """
    dataset_train = PushTLowdimDataset(horizon=16, 
                                       # max_train_episodes=90, # NOTE: u can strict the data
                                       pad_after=7, 
                                       pad_before=1,
                                       seed=seed,
                                       val_ratio=0.02,
                                       zarr_path=data_path)
    dataset_val = dataset_train.get_validation_dataset()
    normalizer = dataset_train.get_normalizer()
    return dataset_train, dataset_val, normalizer

def load_sep_vae_model(vae_local, vae_ckpt_paths):
    """
    @func:
    load vae model separately | [x + y + z + rotation6d + gripper]
    """
    for idx in range(len(vae_ckpt_paths)):
        args = torch.load(vae_ckpt_paths[idx], map_location='cpu')['args']
        vae_local.load_state_dict_sep(torch.load(vae_ckpt_paths[idx], map_location='cpu')['trainer']['vae_wo_ddp'], act_dim=idx, strict=False, using_znorm = args['vqnorm']) # cosine | euler | no need for strict matching
        print(f'[INFO] load vae ckpt {vae_ckpt_paths[idx]}')
    return vae_local

def load_obs_encoder(shape_meta):
    """
    @func:
    """
    # initialize the shape meta
    crop_shape=(76, 76)
    obs_encoder_group_norm=True
    eval_fixed_crop=True
    # parse shape_meta
    action_shape = shape_meta['action']['shape']
    assert len(action_shape) == 1
    action_dim = action_shape[0]
    obs_shape_meta = shape_meta['obs']
    obs_config = {
        'low_dim': [],
        'rgb': [],
        'depth': [],
        'scan': []
    }
    obs_key_shapes = dict()
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        obs_key_shapes[key] = list(shape)
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            obs_config['rgb'].append(key)
        elif type == 'low_dim':
            obs_config['low_dim'].append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {type}")
    # get raw robomimic config
    config = get_robomimic_config(
        algo_name='bc_rnn',
        hdf5_type='image',
        task_name='square',
        dataset_type='ph')
    with config.unlocked():
        # set config with shape_meta
        config.observation.modalities.obs = obs_config
        # set random crop parameter
        ch, cw = crop_shape
        for key, modality in config.observation.encoder.items():
            if modality.obs_randomizer_class == 'CropRandomizer':
                modality.obs_randomizer_kwargs.crop_height = ch
                modality.obs_randomizer_kwargs.crop_width = cw
    # init global state
    ObsUtils.initialize_obs_utils_with_config(config)
    # load model
    policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device='cpu',
        )
    obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
    # replace batch norm with group norm
    if obs_encoder_group_norm:
        replace_submodules(
            root_module=obs_encoder,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features//16, 
                num_channels=x.num_features)
        )
        # obs_encoder.obs_nets['agentview_image'].nets[0].nets
    # obs_encoder.obs_randomizers['agentview_image']
    if eval_fixed_crop:
        replace_submodules(
            root_module=obs_encoder,
            # predicate=lambda x: isinstance(x, obs_core.CropRandomizer), # in multi-task env
            predicate=lambda x: isinstance(x, rmbn.CropRandomizer), # in single-task env
            func=lambda x: dmvc.CropRandomizer(
                input_shape=x.input_shape,
                crop_height=x.crop_height,
                crop_width=x.crop_width,
                num_crops=x.num_crops,
                pos_enc=x.pos_enc
            )
        )
    # already initialized
    return obs_encoder

def load_robomimic_image_runner(output_dir, shape_meta, dataset_path, max_steps=400, n_obs_steps=2, n_action_steps=8, vis_rate=0.0, num_test=50, num_train=6):
    """
    @func:
    """
    env_runner = RobomimicImageRunner(output_dir=output_dir,
                                      dataset_path=dataset_path,
                                      shape_meta = shape_meta,
                                      crf=22,
                                      abs_action=True, # absolute action
                                      max_steps=max_steps,
                                      n_action_steps=n_action_steps,
                                      n_obs_steps=n_obs_steps,          
                                      n_envs=28,
                                      n_test=num_test, # TEST ENV
                                      n_test_vis=int(num_test*vis_rate), # VIS
                                      n_train=num_train, # TRAIN ENV
                                      n_train_vis=int(num_train*vis_rate), # VIS
                                      past_action=False, # futile 
                                      test_start_seed=4600000,
                                      train_start_idx=0,)
    return env_runner

def load_robomimic_lowdim_runner(output_dir, dataset_path, max_steps=400, n_obs_steps=2, n_action_steps=8, vis_rate=0.0, num_test=50, num_train=6):
    """
    @func:
    """
    env_runner = RobomimicLowdimRunner(output_dir=output_dir,
                                       dataset_path=dataset_path,
                                       obs_keys=['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],
                                       crf=22,
                                       abs_action=True, # absolute action
                                       max_steps=max_steps, 
                                       n_action_steps=n_action_steps,
                                       n_envs=28,
                                       n_latency_steps=0,
                                       n_obs_steps=n_obs_steps,
                                       n_test=num_test, # TEST ENV
                                       n_test_vis=int(num_test*vis_rate), # VIS
                                       n_train=num_train, # TRAIN ENV
                                       n_train_vis=int(num_train*vis_rate), # VIS
                                       past_action=False, # futile
                                       render_hw=[128,128],
                                       test_start_seed=4600000,
                                       train_start_idx=0)
    return env_runner

def load_kitchen_lowdim_runner(output_dir, dataset_path, max_steps=280, n_obs_steps=2, n_action_steps=8, vis_rate=0.0, num_test=50, num_train=6):
    """
    @func:
    """
    env_runner = KitchenLowdimRunner(output_dir=output_dir,
                                     abs_action=True, # absolute action
                                     dataset_dir=os.path.dirname(dataset_path),
                                     fps=12.5,
                                     max_steps=max_steps,
                                     n_action_steps=n_action_steps,
                                     n_envs=28, # None
                                     n_obs_steps=n_obs_steps,
                                     n_test=num_test, # TEST ENV
                                     n_test_vis=int(num_test*vis_rate), # VIS
                                     n_train=num_train, # TRAIN ENV
                                     n_train_vis=int(num_train*vis_rate), # VIS
                                     past_action=False, # futile
                                     render_hw=[240,360],
                                     robot_noise_ratio=0.1,
                                     test_start_seed=100000,
                                     train_start_seed=0)
    return env_runner

def load_pusht_lowdim_runner(output_dir, max_steps=300, n_obs_steps=2, n_action_steps=8, vis_rate=0.0, num_test=50, num_train=6):
    """
    :func:
    """
    env_runner = PushTKeypointsRunner(output_dir=output_dir,
                                      agent_keypoints=False,
                                      fps=10,
                                      keypoint_visible_rate=1.0,
                                      legacy_test=True,
                                      max_steps=max_steps,
                                      n_action_steps=n_action_steps,
                                      n_envs=28, # None
                                      n_latency_steps=0,
                                      n_obs_steps=n_obs_steps,
                                      n_test=num_test,
                                      n_test_vis=int(num_test*vis_rate),
                                      n_train=num_train,
                                      n_train_vis=int(num_train*vis_rate),
                                      past_action=False,
                                      test_start_seed=100000,
                                      train_start_seed=0)
    return env_runner

def load_pusht_lowdim_fixed_ini_runner(output_dir, max_steps=300, n_obs_steps=2, n_action_steps=8, vis_rate=0.0, num_test=100):
    """
    :func:
    """
    # goal is [256,256]
    reset_to_state = np.array([
        306, 206,   # agent position
        216, 296,   # block position
        np.pi / 4   # block rotation
        ])
    render_action=False
    env_runner = PushTKeypointsFixedIniRunner(output_dir=output_dir,
                                              agent_keypoints=False,
                                              fps=10,
                                              keypoint_visible_rate=1.0,
                                              legacy_test=False,
                                              max_steps=max_steps,
                                              n_action_steps=n_action_steps,
                                              n_envs=None,
                                              n_latency_steps=0,
                                              n_obs_steps=n_obs_steps,
                                              n_test=num_test,
                                              n_test_vis=int(num_test*vis_rate),
                                              past_action=False,
                                              test_start_seed=100000,
                                              reset_to_state=reset_to_state,
                                              render_action=render_action,)
    return env_runner

def save_runner_json(runner_log,output_dir):
    """
    @func:
    """
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    # json.dump(json_log, open(output_dir, 'w'), indent=2, sort_keys=True)
    with open(output_dir, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)



