import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
from typing import List, Optional, Sequence, Tuple, Union
import dist

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]

class VectorQuantizer2(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        Cvae, 
        using_znorm, 
        beta: float = 0.25,
        default_qresi_counts=0, 
        v_patch_nums=None, 
        quant_resi=0.5, 
        share_quant_resi=4,
    ):
        super().__init__()
        ### params
        self.vocab_size: int = vocab_size # the size of codebook
        self.Cvae: int = Cvae # the channel of vae
        self.using_znorm: bool = using_znorm # whether using Cosine Similarity instead of Euclidean Distance 
        self.v_patch_nums: Tuple[int] = v_patch_nums # the scale of each patch
        ### residual function
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0: # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else: # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        ### codebook usage record
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        ### vqvae loss
        self.beta: float = beta
        ### codebook
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        # initial parameters
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        # run
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums) # total number of depth
            # from small to large
            for si, pn in enumerate(self.v_patch_nums): 
                # find the nearest embedding
                if self.using_znorm:
                    # Cosine Similarity | A*B / (||A|| * ||B||)
                    rest_NC = F.interpolate(f_rest, size=(pn, 1), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # down sample
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # vocab_num
                else:
                    # Euclidean Distance | (A-B)^2
                    rest_NC = F.interpolate(f_rest, size=(pn, 1), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # [B*h*w, vocab_size]
                    idx_N = torch.argmin(d_no_grad, dim=1) # [B*h*w,]
                # hit times
                hit_V = idx_N.bincount(minlength=self.vocab_size).float() # count the hit each vocabulary times in codebook
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True) # calc all of the hit_V value in all processes
                # calc loss
                idx_Bhw = idx_N.view(B, pn, 1) # [B,h,w]
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, 1), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous() # up sample
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # residual function | neural network
                # update
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                # update hit vocab
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                # calculate loss
                vocab_hit_V.add_(hit_V) # hit times calculate
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad) # the loss of vqvae
            # post-process
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw) # discard the grad of f_hat when sent into the decoder
        # margin = pn*1 / 100 | usages is the utilization percentage of the codebook
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)] # usage
        else: usages = None
        # return
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def f_to_idxBl_or_fhat(self, 
                           f_BChw: torch.Tensor, 
                           to_fhat: bool, 
                           v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  
        """
        @func: 
        get the index map of each layer or the final f_hat
        @input: 
        f_BChw has shape [batch_size, cvae, last_patch, last_patch]
        """
        # initial
        B, C, H, W = f_BChw.shape # has shape [B,8,4,1]
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        patch_hws = [(pn, 1) if isinstance(pn, int) else (pn[0], 1) for pn in (v_patch_nums or self.v_patch_nums)] # from small to large | [(pn,1)...]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        SN = len(patch_hws)
        # run
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # z_NC from [batch_size, Cvae, 4, 1] to(area) [batch_size, H, 1, Cvae] to [batch_size*H*1, Cvae]
            if self.using_znorm:
                # Cosine Similarity | A*B / (||A|| * ||B||)
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # z_k
            else:
                # Euclidean Distance | (A-B)^2
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False) # [batch_size, 1] + [vocab_size, ] = [batch_size, vocab_size]
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size) | A^2+B^2-2AB = (A-B)^2
                idx_N = torch.argmin(d_no_grad, dim=1) # (B)
            idx_Bhw = idx_N.view(B, ph, pw) # from [B*ph*pw,] to [B, ph, pw]
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous() # from [B, ph, pw, Cvae] to [B, Cvae, ph, pw] to [B, Cvae, 16, 1]
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h_BChw) # for reconstruction
            f_rest.sub_(h_BChw) # for encoding
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw)) # output
        # return
        return f_hat_or_idx_Bl
    
    def idxBl_to_autoreg_input(self, 
                               gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        @func: 
        only used in CFAP training, for getting teacher-forcing input
        """
        next_scales = [] # 
        B = gt_ms_idx_Bl[0].shape[0] # batch_size
        C = self.Cvae # the channel of vqvae
        H = self.v_patch_nums[-1] # last patch's size
        W = 1 # for action's case
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32) # [batch_size, Cvae, last_patch, last_patch]
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            # z_k from [batch_size, 1, Cvae] to [batch_size, Cvae, 1, 1] to(bicubic) [batch_size, Cvae, 4, 1]
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, 1), size=(H, W), mode='bicubic') # shape->[B,8,4,1]
            # \hat{f} = \hat{f} + \phi_k(z_k)
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw)) # shape->[B,8,4,1]
            pn_next = self.v_patch_nums[si+1]
            # \hat{f} from [batch_size, Cvae, 4, 1] to(area) [batch_size, Cvae, pn_next, 1] to [batch_size, pn_next*1, Cvae]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, 1), mode='area').view(B, C, -1).transpose(1, 2)) 
        return torch.cat(next_scales, dim=1) if len(next_scales) else None # cat BlCs to BLC, this should be float32
    
    def get_next_autoreg_input(self, 
                               si: int, 
                               SN: int, 
                               f_hat: torch.Tensor, 
                               h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in CFAP inference
        """
        @func: 
        only used in CFAP inference, for getting next step's input
        """
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, 1), mode='bicubic')) # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], 1), mode='area') #
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat
    
class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi
    
class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'

class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
