import torch.nn as nn
from typing import Tuple
from MSAT.vqvae import MultiScaleActionTokenizer
from .autoreg import Coarse2FineAutoRegressor

def build_vae_ar(
    ### shared params
    device, 
    patch_nums=(1, 2, 3, 4),
    ### vae args
    V=512, 
    Cvae=8, 
    ch=2, 
    action_dim=10,
    dropout=0.0,
    num_actions=16,
    beta=0.25,
    using_znorm=False,
    quant_conv_ks=3,
    quant_resi=0.5,
    share_quant_resi=4,
    ### ar args
    obs_encoder = None,
    depth=8,
    shared_aln=False, 
    attn_l2_norm=True, 
    init_adaln=0.5, 
    init_adaln_gamma=1e-5, 
    init_head=0.02, 
    init_std=-1, # init_std < 0: automated
    n_obs_steps=2,
    obs_dim=23,
    embed_dim=64,
) -> Tuple[MultiScaleActionTokenizer, Coarse2FineAutoRegressor]:
    """
    @func:

    """
    
    heads = depth # num of heads for ar
    dpr = 0.1 * depth/24 # drop path rate
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, 
                nn.LayerNorm, 
                nn.BatchNorm2d, 
                nn.SyncBatchNorm, 
                nn.Conv1d, 
                nn.Conv2d, 
                nn.ConvTranspose1d, 
                nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    ### Multi-Scale Aaction Tokenizer
    vae_local = MultiScaleActionTokenizer(
        # vocabulary
        vocab_size=V, # 
        # encoder | decoder
        z_channels=Cvae, # 
        ch=ch, # 
        action_dim=action_dim, # 
        num_actions=num_actions, # 
        dropout=dropout, #
        # quant
        beta=beta, # commitment loss weight
        using_znorm=using_znorm, # whether to normalize when computing the nearest neighbors
        quant_conv_ks=quant_conv_ks, # quant conv kernel size
        quant_resi=quant_resi, # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=share_quant_resi, # use 4 \phi layers for K scales: partially-shared \phi
        v_patch_nums=patch_nums, # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        # mode
        test_mode=True,
        ).to(device)
    
    ### Coarse-to-Fine Autoregressive Policy
    ar_wo_ddp = Coarse2FineAutoRegressor(
        vae_proxy=vae_local,
        obs_encoder=obs_encoder,
        action_dim=action_dim,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=heads,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6,
        shared_aln=shared_aln,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        n_obs_steps=n_obs_steps,
        obs_dim=obs_dim,
    ).to(device)
    
    ar_wo_ddp.init_weights(init_adaln=init_adaln, 
                           init_adaln_gamma=init_adaln_gamma, 
                           init_head=init_head, 
                           init_std=init_std)
    
    return vae_local, ar_wo_ddp


