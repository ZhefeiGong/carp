import gc
import os
os.environ["MUJOCO_GL"]="osmesa"
import shutil
import sys
import time
import warnings
from functools import partial
import torch
from torch.utils.data import DataLoader
import dist
from utils import arg_util, misc
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from utils.train_util import load_multiple_robomimic_lowdim_dataset

class NullDDP(torch.nn.Module):
    """
    @func: 
    the ddp for non-distributed
    """
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def build_everything(args: arg_util.Args):
    """
    @func: 
    build everthing for training process
    """
    
    ### resume the model
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    ### create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    ### log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    ### build data    
    print(f'[build PT data] ...\n')
    
    ### load all of the data | actions has shape [act_horizon, act_dim]
    dataset_train, dataset_val, normalizer = load_multiple_robomimic_lowdim_dataset(data_paths=args.data_paths,
                                                                                    data_names=args.data_names,
                                                                                    seed=args.seed,
                                                                                    is_only_act=True) # only load the action for training
    types = str((type(dataset_train).__name__, type(dataset_val).__name__))
    
    ## build the distributed validation dataset
    ld_val = DataLoader(
        dataset_val, # DatasetFolder obj
        num_workers=0,
        pin_memory=True,
        batch_size=round(args.batch_size*1.5),
        # distributed sample
        sampler=EvalDistributedSampler(
            dataset_val, 
            num_replicas=dist.get_world_size(), 
            rank=dist.get_rank()),
        shuffle=False, 
        drop_last=False,
    )
    del dataset_val
    
    ## build the distributed training dataset
    ld_train = DataLoader(
        dataset=dataset_train, # DatasetFolder obj
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        # distributed sample
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), 
            glb_batch_size=args.glb_batch_size, # the global batch size which contains all of the gpus
            same_seed_for_all_ranks=args.same_seed_for_all_ranks, # the seed of all of the ranks
            shuffle=True, 
            fill_last=True, 
            rank=dist.get_rank(), # the rank
            world_size=dist.get_world_size(), # the total number of gpus
            start_ep=start_ep, 
            start_it=start_it, 
        ),
    )
    del dataset_train
    
    [print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    iters_train = len(ld_train)
    ld_train = iter(ld_train)
    # noinspection PyArgumentList
    print(f'[dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    ### build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from trainer_vae import VQVAETrainer
    from optim.amp_opt import AmpOptimizer
    from optim.lr_control import filter_params
    from svqvae import VQVAE, build_vae_disc
    
    ### load vae
    vae_wo_ddp = build_vae_disc(
        device=dist.get_device(),
        ## encoder | decoder
        V=args.vocab_size, 
        Cvae=args.vocab_ch, 
        ch=args.vch, 
        action_dim=1, # here we train a separate vqvae for each dimension
        num_actions=args.act_horizon,
        dropout=args.vdrop,
        ## quant
        beta=args.vqbeta,
        using_znorm=args.vqnorm,
        quant_conv_ks=3, # fixed here
        quant_resi=args.vqresi,
        share_quant_resi=len(args.patch_nums),
        patch_nums=args.patch_nums,
        vae_init=args.vae_init,
        vocab_init=args.vocab_init,
    )
    
    ### load models
    vae_wo_ddp: VQVAE = args.compile_model(vae_wo_ddp, args.vfast)
    assert all(p.requires_grad for p in vae_wo_ddp.parameters())
    
    ### multi-gpu training | load model for each gpu
    vae: DDP = (DDP if dist.initialized() else NullDDP)(vae_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    ### showcase
    print(f'[INIT] VQVAE model = {vae_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_wo_ddp), 
                                                                          ('VAE.enc', vae_wo_ddp.encoder), 
                                                                          ('VAE.dec', vae_wo_ddp.decoder), 
                                                                          ('VAE.quant', vae_wo_ddp.quantizer))]))
    
    ### construct the params for building optimizer
    names, paras, para_groups = filter_params(vae_wo_ddp, nowd_keys={
        'cls_token', 
        'start_token', 
        'task_token', 
        'cfg_uncond',
        'pos_embed', 
        'pos_1LC', 
        'pos_start', 
        'start_pos', 
        'lvl_embed',
        'gamma', 
        'beta',
        'ada_gss', 
        'moe_bias',
        'scale_mul',
        'class_emb', 
        'embedding',
        'norm_scale',
    })
    beta1, beta2 = map(float, args.vopt_beta.split('_'))
    opt_clz = {
        'adamw': partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.afuse,), # torch >= 2.0
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.vlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
    ### build the optimizer
    vae_opt = AmpOptimizer(
        mixed_precision=args.fp16, 
        optimizer=opt_clz(params=para_groups, **opt_kw), 
        names=names,
        paras=paras,
        grad_clip=args.vclip, # whether to utilize the gradient clipping
        n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    ### build trainer
    trainer = VQVAETrainer(
        device = args.device,
        normalizer=normalizer,
        vae_wo_ddp = vae_wo_ddp,
        vae = vae,
        vae_opt = vae_opt,
        ema_ratio = args.vema,
        is_ema = args.vema>0.0,
        act_dim_sep = args.act_dim_sep,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False) # don't load vae again
    del vae_wo_ddp, vae, vae_opt
    
    ### synchronize each distribution
    dist.barrier()
    
    ### retrun the building results
    return (
        tb_lg, # the tensorboard object
        trainer, # the trainer
        start_ep, # the start idx of epoch
        start_it, # the start idx of iteration of a the above epoch
        iters_train, # the total iterative number of training dataloader
        ld_train, # the iterator of training dataloader
        ld_val # the iterator of validation dataloader
    )

def train_one_ep(
        ep: int, 
        is_first_ep: bool, 
        start_it: int, 
        args: arg_util.Args, 
        tb_lg: misc.TensorboardLogger, 
        ld_or_itrt, 
        iters_train: int, 
        trainer):
    """
    @func : 
    train for one epoch
    """
    
    ### import heavy packages after Dataloader object creation
    from trainer_vae import VQVAETrainer
    from optim.lr_control import lr_wd_annealing
    trainer: VQVAETrainer
    
    ### 
    step_cnt = 0
    me_lg = misc.MetricLogger(delimiter='  ')
    me_lg.add_meter('vlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    
    ### current iter | max iter
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    ### run every batch
    for it, obj in me_lg.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        
        ## resume
        g_it = ep * iters_train + it
        if it < start_it: continue # skip the start iter
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        ## move
        actions = obj['action'].to(args.device, non_blocking=True) # non_blocking = asynchronous 
        args.cur_it = f'{it+1}/{iters_train}'
        
        ## set the 'learning rate (lr)' and the 'weight decay (wd)' | for training process
        wp_it = args.vwp * iters_train
        min_vlr, max_vlr, min_vwd, max_vwd = lr_wd_annealing(args.vsche, 
                                                             trainer.vae_opt.optimizer, 
                                                             args.vlr, 
                                                             args.vwd, 
                                                             args.vwde, 
                                                             g_it, 
                                                             wp_it, 
                                                             max_it, 
                                                             wp0=args.vwp0, 
                                                             wpe=args.vwpe)
        args.cur_lr, args.cur_wd = max_vlr, max_vwd # wd(weight decay) is used for the normalization of the weight
        
        ## gradient accumulation for a huge batch training
        stepping = (g_it + 1) % args.ac == 0 # whether finish the gradient update
        step_cnt += int(stepping) # step_cnt update according to the gradient accumulation
        
        ## train one step
        grad_norm, scale_log2 = trainer.train_step(
            it=it, # local iter
            g_it=g_it, # global iter
            stepping=stepping, # whether doing the Gradient Accumulation
            me_lg=me_lg, # metric logger
            tb_lg=tb_lg, # tensorboard logger
            inp=actions, # original input actions
        )
        
        ## write into the tensorboard
        # metric logger update
        me_lg.update(vlr=max_vlr) 
        me_lg.update(grad_norm=grad_norm)
        # tensorboard logger update
        tb_lg.update(head='VAE_opt_lr/lr_min', sche_tlr=min_vlr, step=g_it) # opt_lr
        tb_lg.update(head='VAE_opt_lr/lr_max', sche_tlr=max_vlr, step=g_it) 
        tb_lg.update(head='VAE_opt_wd/wd_max', sche_twd=max_vwd, step=g_it) # opt_wd
        tb_lg.update(head='VAE_opt_wd/wd_min', sche_twd=min_vwd, step=g_it)
        tb_lg.update(head='VAE_opt_grad/fp16', scale_log2=scale_log2, step=g_it) # opt_grad
        tb_lg.update(head='VAE_opt_grad/grad', grad_norm=grad_norm, step=g_it)
        tb_lg.update(head='VAE_opt_grad/grad', grad_clip=args.vclip, step=g_it)
        
    me_lg.synchronize_between_processes()
    
    ## 
    return {k: meter.global_avg for k, meter in me_lg.meters.items()}, me_lg.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost

def main_training():
    """
    @func: 
    the entry for main training process
    """
    
    print("🔥🔥🔥🔥 BEGIN 🔥🔥🔥🔥")
    
    ### get args and update the original args
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    ### build everything
    (
        tb_lg, 
        trainer,
        start_ep, 
        start_it,
        iters_train, 
        ld_train, 
        ld_val
    ) = build_everything(args)
    
    ### params
    start_time = time.time()
    
    ### start to train
    for ep in range(start_ep, args.ep):
        
        ## change the sample randomly in each epoch
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        
        ## current step
        tb_lg.set_step(ep * iters_train)
        
        ## train for one epoch
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, # which epoch are we in
            ep == start_ep, # whether the start epoch
            start_it if ep == start_ep else 0, # which iter are we in of the specific epoch
            args, #
            tb_lg, # tensorboard log
            ld_train, # 
            iters_train, # the total number of each epoch
            trainer # 
        )
        
        ## log / save
        VAE_ep = dict(stats)
        is_val_and_also_saving = (ep + 1) % args.saving_interval == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            ## evaluate the model (discarded here)
            best_updated = False
            ## save ckpt
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, f'vae-ckpt-{ep+1}.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'vae-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'[saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            ## wait all processes
            dist.barrier()
        
        ### log 
        tb_lg.update(head='VAE_ep', step=ep+1, **VAE_ep)
        args.dump_log(); tb_lg.flush()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'[*] [PT finished]  Total cost: {total_time}')
    
    ### refresh
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    
    ### wait all processes
    dist.barrier()

if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()


