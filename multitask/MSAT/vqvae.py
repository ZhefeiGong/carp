from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from svqvae.basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2

class MultiScaleActionTokenizer(nn.Module):
    def __init__(
        self,                   
        # 
        vocab_size=512, # the size of the codebook
        # encoder | decoder
        z_channels=8, # the dimension of the codebook
        ch=2,
        ch_mult=(2, 4),
        action_dim=10,
        num_actions=16,
        dropout=0.0,
        # quant
        beta=0.25, # commitment loss weight
        using_znorm=True, # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3, # quant conv kernel size
        quant_resi=0.5, # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4, # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        # 
        test_mode=True,
    ):
        super().__init__()
        
        ### initialize params
        self.test_mode = test_mode
        self.V = vocab_size
        self.vocab_size = vocab_size
        self.Cvae = z_channels
        
        ### assign to combine each vqvae for each dimension
        self.action_dim = action_dim # 
        self.action_begin_end = list(range(action_dim + 1))  # includes 0 to action_dim
        self.action_in_channels = [1] * action_dim  # all values are 1
        self.action_ch = [ch] * action_dim  # all values are ch (which is 2 as default)
        
        ### x | y | z | rotation6d | gripper
        self.act_feat_dim = num_actions // (2 ** len(ch_mult))
        self.v_patch_nums = v_patch_nums
        self.encoders = nn.ModuleList([
            Encoder(
                ch=self.action_ch[idx], 
                ch_mult=ch_mult, 
                in_channels=self.action_in_channels[idx],
                z_channels=z_channels, 
                action_dim=1,
                num_actions=num_actions, 
                dropout=dropout,) for idx in range(self.action_dim)])
        self.decoders = nn.ModuleList([
            Decoder(
                ch=self.action_ch[idx], 
                ch_mult=ch_mult, 
                in_channels=self.action_in_channels[idx],
                z_channels=z_channels, 
                action_dim=1,
                num_actions=num_actions, 
                dropout=dropout,) for idx in range(self.action_dim)])
        self.quantizers = nn.ModuleList([
            VectorQuantizer2(
                vocab_size=self.vocab_size,
                Cvae=self.Cvae,
                using_znorm=using_znorm, 
                beta=beta,
                default_qresi_counts=default_qresi_counts, 
                v_patch_nums=v_patch_nums, 
                quant_resi=quant_resi, 
                share_quant_resi=share_quant_resi,
            ) for _ in range(self.action_dim)
        ])
        self.quant_convs = nn.ModuleList([nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2) for _ in range(self.action_dim)]) 
        self.post_quant_convs = nn.ModuleList([nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2) for _ in range(self.action_dim)]) 
        
        ### 
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False): # -> rec_BLC, idx_N, loss
        assert self.action_dim == inp.shape[-1], "mismatch the dimension of the actiions"
        outputs = [] 
        usages = []
        vq_losses = []
        for i in range(self.action_dim): 
            inp_slice = inp[:, :, :, self.action_begin_end[i]:self.action_begin_end[i+1]] # -> [B,1,16,1]
            # calculate
            VectorQuantizer2.forward
            f_hat, usage, vq_loss = self.quantizers[i](self.quant_convs[i](self.encoders[i](inp_slice)), ret_usages=ret_usages) # -> [B,8,4,1]
            inp_hat = self.decoders[i](self.post_quant_convs[i](f_hat)) # -> [B,1,16,1]
            # accumulate
            outputs.append(inp_hat)
            vq_losses.append(vq_loss)
            usages.append(usage)
        output = torch.cat(outputs, dim=-1) # -> [B,1,16,action_dim]
        total_vq_loss = torch.sum(torch.stack(vq_losses), dim=0)
        return output, usages, total_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_action(self, f_hat: torch.Tensor):
        """
        @func: 
        transform the latent representaton to actions
        @input:
        f_hat has shape [B,Cvae,last_l,last_w]
        """
        B, Cave, last_l, last_w = f_hat.shape
        action_splits = torch.split(f_hat, 1, dim=2) # in last_l dim
        actions = []
        for idx_act in range(self.action_dim):
            f_hat_act = torch.cat(action_splits[idx_act::self.action_dim], dim=2)  # -> [B,Cvae,last_l//action_dim,1]
            inp_hat = self.decoders[idx_act](self.post_quant_convs[idx_act](f_hat_act))  # -> [B,1,16,1]
            actions.append(inp_hat)
        actions = torch.cat(actions, dim=-1)  # -> [B,1,16,action_dim]
        return actions
    
    def inp_to_action(self, inp_no_grad: torch.Tensor):
        """
        @func: 
        get the reconstruction actions
        """
        assert self.action_dim == inp_no_grad.shape[-1], "mismatch the dimension of the actiions"
        actions = []
        for i in range(self.action_dim): 
            inp_slice = inp_no_grad[:, :, :, self.action_begin_end[i]:self.action_begin_end[i+1]] # [B,1,16,1]
            # calculate
            f = self.quant_convs[i](self.encoders[i](inp_slice))
            f_hat = self.quantizers[i].f_to_idxBl_or_fhat(f, to_fhat=True) # list([B,Cvae,l,l])
            inp_hat = self.decoders[i](self.post_quant_convs[i](f_hat[-1])) # [B,1,num_actions,action_dim] | [B,1,16,1]
            # accumulate
            actions.append(inp_hat)
        actions = torch.cat(actions, dim=-1) # [B,1,16,action_dim]
        return actions
    
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
        idxBls = []
        for i in range(self.action_dim): 
            inp_slice = inp_no_grad[:, :, :, self.action_begin_end[i]:self.action_begin_end[i+1]] 
            f = self.quant_convs[i](self.encoders[i](inp_slice)) # f has shape [batch_size, 8, 4, 1]
            idxBl = self.quantizers[i].f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums) # idxBl has shape [(B,l)]
            idxBls.append(idxBl)
        return idxBls # -> List[List([B,l])]
    
    def idxBl_to_autoreg_input(self, 
                               idxBls:List[List[torch.Tensor]]):
        """
        @func: 
        build the input of autoregressive according to the idxBl
        """
        assert len(idxBls)==self.action_dim, "the dim of gt_idx_bl mismatch with the dime of action"
        inputs = []
        for i in range(self.action_dim):
            inp = self.quantizers[i].idxBl_to_autoreg_input(idxBls[i]) # [B,2+3+4,8]
            inputs.append(inp) 
        return inputs
    
    def idxBl_to_embeddings(self, idxBl):
        """
        @func: 
        get the embeddings of each 
        """
        embeddings = []
        assert idxBl.shape[-1]%self.action_dim==0, "mismatch between the size of idxBl and action_dim"
        for i in range(idxBl.shape[-1]):
            embeddings.append(self.quantizers[i%self.action_dim].embedding(idxBl[:,i:i+1]))
        embeddings = torch.cat(embeddings,dim=1)
        return embeddings
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        """
        @func: 
        get the input for next auto-regressive inference
        """
        B, Cvae, last_l, last_w = f_hat.shape
        f_hat_out, next_token_map_out = [], []
        def collect_and_concat(tensor, idx_act, limit):
            return torch.cat([tensor[:, :, idx_act + idx_si * self.action_dim:idx_act + idx_si * self.action_dim + 1, :] 
                            for idx_si in range(limit)], dim=2)
        for idx_act in range(self.action_dim):
            h_BChw_act = collect_and_concat(h_BChw, idx_act, self.v_patch_nums[si]) # [B,Cvae,pn,1]
            f_hat_act = collect_and_concat(f_hat, idx_act, self.act_feat_dim) # [B,Cvae,act_feat_dim,1]
            f_hat_tmp, next_token_map_tmp = self.quantizers[idx_act].get_next_autoreg_input(si, SN, f_hat_act, h_BChw_act) # [B,Cvae,last_l,1] | [B,Cvae,next_l,1]
            f_hat_out.append(f_hat_tmp)
            next_token_map_out.append(next_token_map_tmp)
        f_hat_out = torch.cat([x.unsqueeze(3) for x in f_hat_out], dim=3).reshape(B, Cvae, -1, last_w) # [B,Cvae,last_l*action_dim,1]
        next_token_map_out = torch.cat([x.unsqueeze(3) for x in next_token_map_out], dim=3).reshape(B, Cvae, -1, last_w) # [B,Cvae,next_l*action_dim,1]
        return f_hat_out, next_token_map_out
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        """
        @func:
        load the model and save to weight
        """
        for i in range(self.action_dim): 
            if f'quantizers.{i}.ema_vocab_hit_SV' in state_dict and state_dict[f'quantizers.{i}.ema_vocab_hit_SV'].shape[0] != self.quantizers[i].ema_vocab_hit_SV.shape[0]:
                state_dict[f'quantizers.{i}.ema_vocab_hit_SV'] = self.quantizers[i].ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) # assign=assign | for pytorch >= 2.1.0
    
    def load_state_dict_sep(self, state_dict: Dict[str, Any], act_dim=0, strict=False, assign=False, using_znorm=True):
        """
        @func:
        load the model of each one specific dimension
        """
        # temporary for different similarity
        self.quantizers[act_dim].using_znorm = using_znorm
        # check | config
        assert (act_dim>=0 and act_dim<self.action_dim), "the appointed dimension of action is out of rage" 
        replace_map = {
            'encoder': 'encoders',
            'decoder': 'decoders',
            'quantizer': 'quantizers',
            'quant_conv': 'quant_convs',
            'post_quant_conv': 'post_quant_convs'
        }
        for key in list(state_dict.keys()):
            for old_key, new_prefix in replace_map.items():
                if old_key in key:
                    new_key = key.replace(old_key, f'{new_prefix}.{act_dim}')
                    state_dict[new_key] = state_dict.pop(key)
                    break
        # hit
        if f'quantizers.{act_dim}.ema_vocab_hit_SV' in state_dict and state_dict[f'quantizers.{act_dim}.ema_vocab_hit_SV'].shape[0] != self.quantizers[act_dim].ema_vocab_hit_SV.shape[0]:
            state_dict[f'quantizers.{act_dim}.ema_vocab_hit_SV'] = self.quantizers[act_dim].ema_vocab_hit_SV
        # load | return
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign) # assign=assign | for pytorch >= 2.1.0



