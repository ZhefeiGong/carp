from typing import List, Optional, Tuple, Union
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import dist
from optim.amp_opt import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
from svqvae import VQVAE, VectorQuantizer2
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
from env.common.normalizer import LinearNormalizer

class VQVAETrainer(object):
    def __init__(
        self, 
        device, 
        normalizer: LinearNormalizer,
        vae_wo_ddp: VQVAE,
        vae: DDP,
        vae_opt: AmpOptimizer,
        ema_ratio: float,
        is_ema: bool,
        act_dim_sep: int,
        act_dim_names: List[str],
    ):  
        
        super(VQVAETrainer, self).__init__()
        
        ### models - vae
        self.vae, self.vae_opt = vae, vae_opt
        self.vae_wo_ddp: VQVAE = vae_wo_ddp
        self.vae_params: Tuple[nn.Parameter] = tuple(self.vae_wo_ddp.parameters())
        self.act_dim_sep = act_dim_sep
        
        ### ema for vae
        self.ema_ratio = ema_ratio
        self.is_ema = is_ema
        if self.is_ema:
            self.vae_ema: VQVAE = deepcopy(vae_wo_ddp).eval()
        else:
            self.vae_ema: VQVAE = None
        
        ### normalizer
        self.vae_norm = normalizer
        self.vae_norm.to(device)
        
        ### params - vae
        self.w_l1=0.0 # discarded
        self.w_l2=1.0 # 
        self.w_vq=1.0 # 0.25(smaller) | 1.00(normal)
        self.w_lp=0.0 # discarded
        
        ### usage name for recording
        self.usage_names = act_dim_names
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        pass
    
    def train_step(
        self, 
        it: int, 
        g_it: int, 
        stepping: bool,
        me_lg: MetricLogger, 
        tb_lg: TensorboardLogger,
        inp: FTen, # BLC
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        
        is_loggable = (g_it == 0 or (g_it + 1) % 1 == 0) # recording or not
        
        ### VAE
        # automatic mixed precision
        with self.vae_opt.amp_ctx:
            
            # data initialization
            inp = self.vae_norm['action'].normalize(inp)
            inp = inp.view(inp.shape[0],1,inp.shape[1],inp.shape[2]).contiguous() # [bacth_size, num_channel, num_horizon, num_dimension]
            
            # forward
            self.vae_wo_ddp.forward
            inp_slice = inp[:,:,:,self.act_dim_sep:self.act_dim_sep+1]
            rec_inp_slice, usages, loss_vq = self.vae(inp=inp_slice, ret_usages=is_loggable)
            
            # calc loss
            loss_rec_l1 = F.l1_loss(input=rec_inp_slice, target=inp_slice)
            loss_rec_l2 = F.mse_loss(input=rec_inp_slice, target=inp_slice)

            # combine loss
            loss_vae = self.w_l2 * loss_rec_l2 + self.w_vq * loss_vq
        
        ### VAE Backward
        grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss_vae, stepping=stepping)
        
        ### UPDATE | EMA
        if stepping:
            if self.is_ema:
                self.ema_update(g_it)
        
        ### LOG to metric
        if it == 0 or it in me_lg.log_iters:
            me_lg.update(
                loss_rec_l1=loss_rec_l1.item(), 
                loss_rec_l2=loss_rec_l2.item(), 
                loss_vq=loss_vq.item(), 
                loss_vae=loss_vae.item())
        
        ### LOG to tensorboard
        if is_loggable:
            # loss
            tb_lg.update(
                head='VAE_iter_loss',
                loss_rec_l1=loss_rec_l1.item(), 
                loss_rec_l2=loss_rec_l2.item(), 
                loss_vq=loss_vq.item(), 
                loss_vae=loss_vae.item(),
                step=g_it)
            # usage
            name = self.usage_names[self.act_dim_sep]
            tb_lg.update(head=f"VAE_vocab_usage_{name}",
                        scale_1=usages[0],
                        scale_2=usages[1],
                        scale_3=usages[2],
                        scale_4=usages[3],
                        scale_all=usages[-1],
                        step=g_it)
        
        return grad_norm, scale_log2
    
    def ema_update(self, g_it):
        """
        @func: 
        ema update in order to get a more stable version
        """
        ## init
        ema_ratio = min(self.ema_ratio, (g_it//2 + 1) / (g_it//2 + 10))
        ## params
        for p_ema, p in zip(self.vae_ema.parameters(), self.vae_wo_ddp.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(ema_ratio).add_(p.data, alpha=1-ema_ratio)
        ## buffer
        for p_ema, p in zip(self.vae_ema.buffers(), self.vae_wo_ddp.buffers()):
            p_ema.data.copy_(p.data)
        ## codebook
        quant, quant_ema = self.vae_wo_ddp.quantizer, self.vae_ema.quantizer
        quant: VectorQuantizer2
        if hasattr(quant, 'using_ema') and quant.using_ema: # then embedding.weight requires no grad, thus is not in self.vae_ema_params; so need to update it manually
            if hasattr(quant, 'using_restart') and quant.using_restart:
                # cannot use ema, cuz quantize.embedding uses replacement (rand restart)
                quant_ema.embedding.weight.data.copy_(quant.embedding.weight.data)
            else:
                quant_ema.embedding.weight.data.mul_(ema_ratio).add_(quant.embedding.weight.data, alpha=1-ema_ratio)
    
    def get_config(self):
        """
        @func: 
        get the loss and ema config of the model
        """
        return {
            'ema_ratio': self.ema_ratio,
            'w_l1': self.w_l1,
            'w_l2': self.w_l2, 
            'w_vq': self.w_vq, 
            'w_lp': self.w_lp,
        }
    
    def state_dict(self):
        """
        @func: 
        fetch the models needed to reserve
        """
        state = {'config': self.get_config()}
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt', 'vae_norm'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True):
        """
        @func: 
        load the models needed
        """
        for k in ('vae_wo_ddp', 'vae_ema', 'vae_opt', 'vae_norm'):
            m = getattr(self, k)
            if m is not None:
                
                # model
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                
                # load_state_dict
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETr.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETr.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAETr.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)


