import torch.nn as nn
from typing import Tuple
from .quant import VectorQuantizer2
from .vqvae import MultiScaleActionTokenizer

def build_vae_disc(
    # shared args
    device, 
    # encoder | decoder
    V=512, 
    Cvae=8, 
    ch=2, 
    ch_mult=(2, 4),
    action_dim=10,
    num_actions=16,
    dropout=0.0,
    # quant
    beta=0.25,
    using_znorm=True,
    quant_conv_ks=3,
    quant_resi=0.5,
    share_quant_resi=4,
    patch_nums=(1, 2, 3, 4),
    # initialize
    vae_init=-0.5,
    vocab_init=-1,
) -> Tuple[MultiScaleActionTokenizer, None]:
    """
    @func: 
    initial the vae
    """
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, 
                nn.Embedding,
                nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm, 
                nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                ):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    ### build vae - model
    vae_wo_ddp = MultiScaleActionTokenizer(
        # vocabulary
        vocab_size=V,
        # encoder | decoder
        z_channels=Cvae,
        ch=ch, 
        ch_mult=ch_mult,
        action_dim=action_dim,
        num_actions=num_actions,
        dropout=dropout,
        # quant
        beta=beta, # commitment loss weight
        using_znorm=using_znorm, # whether to normalize when computing the nearest neighbors
        quant_conv_ks=quant_conv_ks, # quant conv kernel size
        quant_resi=quant_resi, # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=share_quant_resi, # use 4 \phi layers for K scales: partially-shared \phi
        v_patch_nums=patch_nums, # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        # 
        test_mode=False,
        ).to(device)
    
    ### initialize the model weight
    vae_init_modules=[
        vae_wo_ddp.encoders, 
        vae_wo_ddp.quant_convs,
        vae_wo_ddp.quantizers,
        vae_wo_ddp.post_quant_convs,
        vae_wo_ddp.decoders,
        ]
    
    ### initialize each separate action vqvae
    for modules in vae_init_modules:
        for i in range(action_dim):
            init_weights(modules[i], vae_init, i)
    for i in range(action_dim):
        init_vocab(vae_wo_ddp.quantizers[i], vocab_init, i)
    return vae_wo_ddp

def init_vocab(quantizer, init: float, idx: int):
    """
    @func : 
    """
    print(f'[init-weights-vocab.{idx}] {type(quantizer.embedding).__name__} with {"std" if init > 0 else "gain"}={abs(init):g}')
    if init > 0:
        nn.init.trunc_normal_(quantizer.embedding.weight.data, std=init)
    elif init < 0:
        base = quantizer.Cvae ** -0.5
        base /= 36
        quantizer.embedding.weight.data.uniform_(-abs(init) * base, abs(init) * base)

def init_weights(model, conv_std_or_gain, idx: int):
    """
    @func : 
    """
    print(f'[init-weights-vae.{idx}] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None: nn.init.constant_(m.weight.data, 1.)



