import math
import torch
import torch.nn as nn
from MSAT.vqvae import MultiScaleActionTokenizer
from functools import partial
from typing import Optional, Tuple, Union, List
from CFAP.basic_ar import AdaLNBeforeHead, AdaLNSelfAttn
from env.common.pytorch_util import dict_apply

class SharedAdaLin(nn.Linear):
    """
    @func: 
    the shared conditional adaptive linear layer
    """
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6 # 6 params for conditional inferring
        return super().forward(cond_BD).view(-1, 1, 6, C) # [B，1，6，C]

class Coarse2FineAutoRegressor(nn.Module):
    def __init__(
        self, 
        vae_proxy: MultiScaleActionTokenizer,
        obs_encoder=None,
        action_dim=10, 
        depth=8, 
        embed_dim=64, 
        num_heads=8, 
        mlp_ratio=4., 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        norm_eps=1e-6, 
        shared_aln=False, 
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4),
        n_obs_steps=1,
        obs_dim=23,
    ):
        
        super().__init__()
        
        #### action dimension
        self.action_dim = action_dim
        
        #### observation encoder | borrow from `hybrid policy` of robomimic
        self.obs_encoder = obs_encoder
        self.n_obs_steps = n_obs_steps
        self.obs_dim = obs_dim
        
        #### hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_proxy.Cvae, vae_proxy.vocab_size
        self.depth, self.num_heads = depth, num_heads
        self.C = embed_dim
        self.D = embed_dim
        self.patch_nums: Tuple[int] =  [x * self.action_dim for x in patch_nums]
        self.L = sum(pn * 1 for pn in self.patch_nums) # the total num of patches
        self.first_l = self.patch_nums[0] * 1 # the length of the first layer
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn * 1))
            cur += pn * 1
        self.num_stages_minus_1 = len(self.patch_nums) - 1 # number of pathes - 1 
        
        #### input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C) # from z_channel to embed_dim
        
        #### observation embedding | condition on obs
        init_std = math.sqrt(1 / self.C / 3)
        self.obs_embed = nn.Linear(self.obs_dim * self.n_obs_steps, self.C) # from obs_dim to C | observation embedding
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C)) # [first_layer, C] | position encoding for the first layer
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std) # "gaussian distribution" for pos_start
        
        #### absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*1, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1) # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC) # position embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C) # [num_scale, C]
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std) # [num_scale, C]
        
        #### backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity() # shared adaptive layer for conditional information
        norm_layer = partial(nn.LayerNorm, eps=norm_eps) # normalization layer
        self.drop_path_rate = drop_path_rate 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, 
                shared_aln=shared_aln, 
                block_idx=block_idx, 
                embed_dim=self.C, 
                norm_layer=norm_layer, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx], 
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(depth)
        ])
        print(
            f'[AR-config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'[AR-drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})\n',
            end='\n\n', 
            flush=True
        )
        
        #### attention mask used in training (for masking out the future) | it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*1,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1) # 1L1
        dT = d.transpose(1, 2) # 11L
        lvl_1L = dT[:, 0].contiguous() # 1L
        self.register_buffer('lvl_1L', lvl_1L) # 1L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L) # 11LL
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous()) # 11LL
        
        #### classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer) 
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, 
                   h: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                   cond_BD: Optional[torch.Tensor]):
        """
        @func: 
        logical layer outputting the vocab's size probability
        """
        return self.head(self.head_nm(h.float(), cond_BD).float()).float() # from C to V | BLV
    
    @torch.no_grad()
    def autoregressive_infer_cfg(self, 
                                 nobs: Optional[List[Union[int, torch.LongTensor]]],
                                 vae_proxy: MultiScaleActionTokenizer,
                                 ) -> torch.Tensor: 
        """
        @func: 
        only used for inference, on autoregressive mode
        @return: 
        if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # encode the observations
        To = self.n_obs_steps
        if self.obs_encoder is not None: # Image-Based
            assert len({nobs[key].shape[0] for key in nobs}) == 1, "get different batch sizes in nobs"
            B = {nobs[key].shape[0] for key in nobs}.pop()
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # [B*To,obs_dim]
            nobs_features = self.obs_encoder(this_nobs).reshape(B, -1) # [B,To*obs_dim]
        else: # State-Based
            B = nobs.shape[0]
            nobs_features = nobs[:,:To,...].reshape(B,-1) # [B,To*obs_dim]
        # params
        sos = cond_BD = self.obs_embed(nobs_features) # sos/cond_BD has shape [batch_size, embed_dim]
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC # [1,L,embed_dim]+[1,L,embed_dim]=[1,L,embed_dim]
        next_token_map = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l] # [B,first_l,embed_dim]
        cur_L = 0 
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], 1) # [B,Cvae,pn_last,1]
        # enable kv caching
        for b in self.blocks: b.attn.kv_caching(True)
        # inference
        for si, pn in enumerate(self.patch_nums): # si: i-th segment
            cur_L += pn*1
            cond_BD_or_gss = self.shared_ada_lin(cond_BD) # [B,embed_dim] or [B,1,6,embed_dim]
            x = next_token_map # [B,self.patch_nums[si]*1,embed_dim]
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD) # [B,self.patch_nums[si]*1,V]
            idx_Bl = logits_BlV.data.argmax(dim=-1) # maximize
            h_BChw = vae_proxy.idxBl_to_embeddings(idx_Bl) # [B, self.patch_nums[si]*1, Cvae]
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, 1) # [B, Cvae, pn, 1]
            f_hat, next_token_map = vae_proxy.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw) # [B,Cvae,last_pn,1] | [B,Cvae,next_pn,1]
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2) # [B,next_pn*1,Cvae]
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] * 1] # [B,next_pn*1,embed_dim]
        # disable kv caching
        for b in self.blocks: b.attn.kv_caching(False) 
        return vae_proxy.fhat_to_action(f_hat)
    
    def forward(self,
                nobs: torch.LongTensor, 
                x_BLCv_wo_first_l: torch.Tensor,
                ) -> torch.Tensor:
        """
        @input:
        :param label_B: label_B has shape [batch_size, ]
        :param x_BLCv_wo_first_l: teacher forcing input [batch_size, total_num_patches_exept_first, self.Cvae]
        @return:
        :logits BLV, V is vocab_size
        """
        ### params
        bg, ed = 0, self.L # ed is L(70)
        B = x_BLCv_wo_first_l.shape[0] # get batch size
        To = self.n_obs_steps
        with torch.cuda.amp.autocast(enabled=False):
            ### encode the observations
            if self.obs_encoder is not None: # Image-Based
                this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) # [B*To,obs_dim]
                nobs_features = self.obs_encoder(this_nobs).reshape(B, -1) # [B,To*obs_dim]
            else: # State-Based
                nobs_features = nobs[:,:To,...].reshape(B,-1) # [B,To*obs_dim]
            ### add observation embedding 
            sos = cond_BD = self.obs_embed(nobs_features) # sos/cond_BD has shape [batch_size, embed_dim] | [B, embed_dim]
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) # [B,first_l,embed_dim] + [B,first_l,embed_dim] = [B,first_l,embed_dim]
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) # [B,first_l,embed_dim] + [B,L-first_l,embed_dim] = [B, L, embed_dim]
            ### add level and position embeddings to x_BLC 
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # NC(BL) + 1Lembed_dim = BLembed_dim + 1Lembed_dim
        ### get bias
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed] # [1, 1, L, L] | [1,1,10*action_dim,10*action_dim]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # whether using a shared adaptive linear : [B, embed_dim] or [B,1,6,embed_dim]
        ### get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8) # [8,8]'s 1
        main_type = torch.matmul(temp, temp).dtype # torch.float16
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        ### forward calculate
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias) # always condition on "cond_BD"
        x_BLV = self.get_logits(x_BLC.float(), cond_BD) # from C to V ｜ BLV
        ### logits BLV, V is vocab_size
        return x_BLV
    
    def init_weights(self, 
                     init_adaln=0.5, 
                     init_adaln_gamma=1e-5, 
                     init_head=0.02, 
                     init_std=0.02, 
                     conv_std_or_gain=0.02):
        """
        @func: 
        initialize the weights
        """
        
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')

        for m in self.modules():
            # No need to initialize the encoder
            if m is self.obs_encoder: 
                continue 
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: AR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
    
