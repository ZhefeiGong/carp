import os
import cv2
import torch
import random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch import nn as nn
from torch.nn import functional as F


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz torch.multinomial can only be used for 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def gumbel_softmax_with_rng(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, rng: torch.Generator = None) -> torch.Tensor:
    if rng is None:
        return F.gumbel_softmax(logits=logits, tau=tau, hard=hard, eps=eps, dim=dim)
    
    gumbels = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=rng).log())
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'


def load_trajectories_from_folder(folder_path):
    trajectories = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            traj = np.load(file_path)  # shape is (T, 2) or (T, D)
            if traj.ndim >= 2 and traj.shape[1] >= 2:
                traj_2d = traj[:, :2]  # only x, y
                trajectories.append(traj_2d)
    return trajectories


def draw_traj_pic(trajectories, save_path, background_path=None, vis_len=40):
    # initialize parameters
    def scale(value, img_dim, data_min=0.0, data_max=512.0):
        return int((value - data_min) / (data_max - data_min) * (img_dim - 1))
    # w/ background
    img_dim=512
    plt.figure(figsize=(img_dim, img_dim), dpi=1)
    ax = plt.gca()
    if background_path is not None:
        background = plt.imread(background_path)
        if background.shape[0] != img_dim or background.shape[1] != img_dim:
            from PIL import Image
            background = Image.fromarray((background * 255).astype(np.uint8))
            background = background.resize((img_dim, img_dim))
            background = np.asarray(background).astype(np.float32) / 255.0
        ax.imshow(background, extent=[0, img_dim, img_dim, 0])  # extent -> set axis rage
    else:
        ax.imshow(np.ones((img_dim, img_dim, 3)), extent=[0, img_dim, img_dim, 0])
    # set
    plt.xlim(0, img_dim)
    plt.ylim(0, img_dim)
    ax.invert_yaxis()
    # draw trajectory
    for trajectory_raw in trajectories:
        trajectory = trajectory_raw[:vis_len]
        cmap = cm.get_cmap('plasma')
        norm = colors.Normalize(vmin=0, vmax=len(trajectory) - 1)
        for i in range(len(trajectory) - 1):
            x1, y1 = scale(trajectory[i][0], img_dim), scale(trajectory[i][1], img_dim)
            x2, y2 = scale(trajectory[i+1][0], img_dim), scale(trajectory[i+1][1], img_dim)
            color = cmap(norm(i))
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=100)
    # save picture
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)


def extract_first_frame_from_random_video(folder_path, trj_bkg_path):
    mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    if not mp4_files:
        raise FileNotFoundError("No .mp4 files found in the specified directory.")
    selected_video = random.choice(mp4_files)
    video_path = os.path.join(folder_path, selected_video)
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Failed to read the first frame from {selected_video}")
    cv2.imwrite(trj_bkg_path, frame)

