from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy

from env.common.pytorch_util import dict_apply
from env.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from env.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from env.common.rotation_transformer import RotationTransformer
from env.common.replay_buffer import ReplayBuffer
from env.common.sampler import (
    SequenceSampler, 
    get_val_mask, 
    downsample_mask)
from env.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            is_only_act=False # only True when training multi-scale action tokenizer
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)
        
        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    is_only_act=is_only_act)
                replay_buffer.add_episode(episode)
        
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.is_only_act = is_only_act
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        # action
        stat = array_to_stats(self.replay_buffer['action']) # ([L,C])=[C,]
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # observation
        if self.is_only_act is False:
            obs_stat = array_to_stats(self.replay_buffer['obs'])
            normalizer['obs'] = normalizer_from_stat(obs_stat)
        
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs) # scale
    offset = np.zeros_like(stat['max']) # 0
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer, is_only_act=False):
    
    if is_only_act is False:
        obs = np.concatenate([
            raw_obs[key] for key in obs_keys
        ], axis=-1).astype(np.float32)
    
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    if is_only_act is False:
        data = {
            'obs': obs,
            'action': raw_actions
        }
    else:
        data = {
            'action': raw_actions
        }

    return data



if __name__ == "__main__":
    
    # dataset_path = "/liujinxin/zhefei/ARGen4IL/workspace/data/robomimic/datasets/square/ph/low_dim_abs.hdf5"
    dataset_path = "/liujinxin/zhefei/ARGen4IL/workspace/data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5"
    dataset = RobomimicReplayLowdimDataset(abs_action=False, dataset_path=dataset_path, horizon=16, pad_after=7, pad_before=1, seed=42, val_ratio=0.02)
    print(len(dataset))
    obs = dataset[0]['obs']
    action = dataset[0]['action']
    print(obs.shape)
    print(action.shape)
    print(action)
    
# python /liujinxin/zhefei/ARGen4IL/workspace/autoencoder/robodata/robomimic_replay_lowdim_dataset.py


