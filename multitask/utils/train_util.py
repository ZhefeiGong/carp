import torch
import wandb
import json
import os
import torch.nn as nn
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as obs_core
import env.model.crop_randomizer as dmvc
from env.common.robomimic_config_util import get_robomimic_config
from env.common.pytorch_util import dict_apply, replace_submodules
from env.runner.robomimic_image_runner import RobomimicImageRunner
from env.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from env.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from env.dataset.robomimic_multi_image_datatset_wrapper import MultiImageDatasetWrapper
from env.dataset.robomimic_multi_lowdim_datatset_wrapper import MultiLowdimDatasetWrapper

def load_shape_meta():
    """
    @func:
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

def load_multiple_robomimic_image_dataset(data_paths, data_names, seed, shape_meta, n_obs_steps):
    """
    @func:
    """
    train_datasets = {}
    val_datasets = {}
    for idx, data_path in enumerate(data_paths):
        dataset_train, dataset_val, normalizer = load_robomimic_image_dataset(data_path, seed, shape_meta, n_obs_steps)
        train_datasets[data_names[idx]] = dataset_train
        val_datasets[data_names[idx]] = dataset_val
    ### wrap
    dataset_train = MultiImageDatasetWrapper(train_datasets)
    dataset_val = MultiImageDatasetWrapper(val_datasets)
    ### normalizer
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

def load_multiple_robomimic_lowdim_dataset(data_paths, data_names, seed, is_only_act=False):
    """
    @func:
    """
    train_datasets = {}
    val_datasets = {}
    for idx, data_path in enumerate(data_paths):
        dataset_train, dataset_val, _ = load_robomimic_lowdim_dataset(data_path, seed, is_only_act)
        train_datasets[data_names[idx]] = dataset_train
        val_datasets[data_names[idx]] = dataset_val
    ### wrap
    dataset_train = MultiLowdimDatasetWrapper(train_datasets)
    dataset_val = MultiLowdimDatasetWrapper(val_datasets)
    ### normalizer
    normalizer = dataset_train.get_normalizer(is_only_act)
    return dataset_train, dataset_val, normalizer

def load_sep_vae_model(vae_local, vae_ckpt_paths):
    """
    @func:
    load vae model separately | [x + y + z + rotation6d + gripper]
    """
    for idx in range(len(vae_ckpt_paths)):
        args = torch.load(vae_ckpt_paths[idx], map_location='cpu')['args']
        vae_local.load_state_dict_sep(torch.load(vae_ckpt_paths[idx], map_location='cpu')['trainer']['vae_wo_ddp'], act_dim=idx, strict=False, using_znorm = args['vqnorm']) # cosine | euler | no need for strict matching
        print('[INFO] load')
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
            predicate=lambda x: isinstance(x, obs_core.CropRandomizer),
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

def load_robomimic_image_runner(output_dir, shape_meta, dataset_path, max_steps, n_obs_steps, n_action_steps, vis_rate=0.0):
    """
    @func:
    """
    num_test = 50
    num_train = 6
    env_runner = RobomimicImageRunner(output_dir=output_dir,
                                      dataset_path=dataset_path,
                                      shape_meta = shape_meta,
                                      crf=22,
                                      abs_action=True,
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



