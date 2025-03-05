from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2

class VQVAE(nn.Module):
    def __init__(
        self,
        # 
        vocab_size=512, # 
        # encoder | decoder
        z_channels=8, # 
        ch=2, # 
        action_dim=1, # 
        num_actions=16, # 
        dropout=0.0,
        # quant
        beta=0.25, # commitment loss weight
        using_znorm=True, # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3, # quant conv kernel size
        quant_resi=0.5, # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4, # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True, # 
    ):
        super().__init__()
        
        self.test_mode = test_mode
        self.V = vocab_size
        self.vocab_size = vocab_size
        self.Cvae = z_channels
        self.action_dim = action_dim
        
        ddconfig = dict(            
            ch=ch, 
            ch_mult=(2, 4), 
            in_channels=1,
            z_channels=z_channels, 
            action_dim=1,
            num_actions=num_actions, 
            dropout=dropout,
        )
        self.downsample = 2 ** (len(ddconfig['ch_mult']))
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantizer: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm, 
            beta=beta,
            default_qresi_counts=default_qresi_counts, 
            v_patch_nums=v_patch_nums, 
            quant_resi=quant_resi, 
            share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2) 
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):   # -> rec_BLC, idx_N, loss
        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantizer(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages) # [B,8,4,1]
        inp_hat = self.decoder(self.post_quant_conv(f_hat)) # [B,1,16,1]
        return inp_hat, usages, vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_action(self, f_hat: torch.Tensor):
        """
        @func: 
        transform the latent representaton to actions
        """
        return self.decoder(self.post_quant_conv(f_hat))  # B,1,16,act_dim
    
    def inp_to_action(self, inp_no_grad: torch.Tensor):
        """
        @func: 
        get the reconstruction actions
        """
        f = self.quant_conv(self.encoder(inp_no_grad))
        f_hat = self.quantizer.f_to_idxBl_or_fhat(f, to_fhat=True) # list([B,Cvae,l,l])
        inp_hat = self.decoder(self.post_quant_conv(f_hat[-1])) # [B,1,num_actions,action_dim] | [B,1,16,1]
        return inp_hat
    
    def inp_to_idxBl(self, 
                     inp_no_grad: torch.Tensor, 
                     v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]: 
        """
        @input: 
        inp_img_no_grad has shape [batch_size, 1, num_actions, action_dim]
        v_patch_nums has shape [] / None
        @return: 
        List[Bl]
        """
        f = self.quant_conv(self.encoder(inp_no_grad)) # f has shape [batch_size, 8, 4, 1]
        idxBl = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums) # idxBl has shape []
        return idxBl
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False, using_znorm = False):

        """
        @func:
        load the model and save to weight
        """ 
        self.quantizer.using_znorm = using_znorm
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) # assign=assign | for pytorch >= 2.1.0



