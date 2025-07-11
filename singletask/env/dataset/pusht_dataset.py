from typing import Dict
import torch
import numpy as np
import copy
from env.common.pytorch_util import dict_apply
from env.common.replay_buffer import ReplayBuffer
from env.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from env.common.normalizer import LinearNormalizer
from env.dataset.base_dataset import BaseLowdimDataset

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

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

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer) # {'obs', 'action'}
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # ### visualization
        # task_name = 'data_distribution'
        # name=['x','y','z','r1','r2','r3','r4','r5','r6','gripper']
        # act_dim = data['action'].shape[-1]
        # import matplotlib.pyplot as plt
        # for i in range(act_dim):
        #     data_x = data['action'][:,i]
        #     plt.hist(data_x, bins=100, edgecolor='black')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.title('Histogram of Data')
        #     plt.savefig(f"tmp/{task_name}/{name[i]}_histogram_raw.png")    
        #     plt.close()
        #     data_x_ = normalizer.normalize(data)['action'][:,i]
        #     plt.hist(data_x_.detach().numpy(), bins=100, edgecolor='black')
        #     plt.xlabel('Value')
        #     plt.ylabel('Frequency')
        #     plt.title('Histogram of Data')
        #     plt.savefig(f"tmp/{task_name}/{name[i]}_histogram_norm.png")    
        #     plt.close()
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs,                             # T, D_o
            'action': sample[self.action_key],      # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
