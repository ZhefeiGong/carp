from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm

from env.common.pytorch_util import dict_apply
from env.common.replay_buffer import ReplayBuffer
from env.common.sampler import SequenceSampler, get_val_mask
from env.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from env.dataset.base_dataset import BaseLowdimDataset
from env.kitchen.kitchen_util import parse_mjl_logs

class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0,
            is_only_act=False # only True when training multi-scale action tokenizer
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)
        
        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                if is_only_act is False:
                    # load both observations and actions
                    qpos = data['qpos'].astype(np.float32)
                    obs = np.concatenate([
                        qpos[:,:9],
                        qpos[:,-21:],
                        np.zeros((len(qpos),30),dtype=np.float32)
                    ], axis=-1)
                    if robot_noise_ratio > 0:
                        # add observation noise to match real robot
                        noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                            low=-1., high=1., size=(obs.shape[0], 30))
                        obs[:,:30] += noise
                    episode = {
                        'obs': obs,
                        'action': data['ctrl'].astype(np.float32)
                    }
                else:
                    # load only actions
                    episode = {
                        'action': data['ctrl'].astype(np.float32)
                    }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
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
    
    def get_normalizer(self, mode='limits', **kwargs):
        if self.is_only_act is False:
            data = {
                'obs': self.replay_buffer['obs'], # (137553, 60)
                'action': self.replay_buffer['action'] # (137553, 9)
            }
        else:
            data = {
                'action': self.replay_buffer['action'] # (137553, 9)
            }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # ### visualization
        # task_name = 'data_distribution'
        # name=['x','y','z','r1','r2','r3','r4','g1','g2']
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
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
