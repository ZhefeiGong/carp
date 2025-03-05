import torch
from typing import Dict
import numpy as np
from typing import Dict, List
import torch
import numpy as np
from env.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from env.dataset.base_dataset import LinearNormalizer
from env.common.normalizer import LinearNormalizer
from env.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class MultiImageDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, datasets: Dict[str, RobomimicReplayImageDataset]):
        """
        :func: 
        """
        self.datasets = datasets
        self.dataset_lengths = {name: len(ds) for name, ds in datasets.items()}
        self.cumulative_lengths = np.cumsum(list(self.dataset_lengths.values()))
        self.total_length = sum(self.dataset_lengths.values())
        
        #
        self.lowdim_keys = next(iter(datasets.values())).lowdim_keys
        self.rgb_keys = next(iter(datasets.values())).rgb_keys
    
    def __len__(self):
        """
        :func:
        """
        return self.total_length
    
    def __getitem__(self, idx):
        """
        :func: 
        """
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side="right")   # index of the dataset
        dataset_name = list(self.datasets.keys())[dataset_idx]                      # name of the dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        data = self.datasets[dataset_name][local_idx]
        data['task'] = torch.tensor(dataset_idx, dtype=torch.long) # long | 🔥 🔥 🔥 | size [B] for each iter
        return data
    
    def combine_replay_buffer(self, name):
        """
        :func:
        """
        replay_buffers = []
        for key, value in self.datasets.items():
            replay_buffers.append(value.replay_buffer[name])
        replay_buffers = np.vstack(replay_buffers)
        return replay_buffers
    
    def vis_act_distribution(self, dataset_name):
        """
        :func:
        """
        ### get data 
        stat = array_to_stats(self.datasets[dataset_name].replay_buffer['action'])
        this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
        ### visualization
        name=['x','y','z','r1','r2','r3','r4','r5','r6','gripper']
        import matplotlib.pyplot as plt
        for i in range(10):
            data_x = self.datasets[dataset_name].replay_buffer['action'][:,i]
            plt.hist(data_x, bins=100, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Data')
            plt.savefig(f"tmp/distribution/{dataset_name}/{name[i]}_histogram_raw.png")    
            plt.close()                     
            original_array = self.datasets[dataset_name].replay_buffer['action'][:,:10]
            data_x_ = this_normalizer.normalize(original_array)[:,i]
            plt.hist(data_x_.detach().numpy(), bins=100, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Data')
            plt.savefig(f"tmp/distribution/{dataset_name}/{name[i]}_histogram_minmax.png")    
            plt.close()
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        :func:
        """
        # init
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.combine_replay_buffer('action'))
        this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.combine_replay_buffer(key))

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # return
        return normalizer


if __name__ == '__main__':
    pass