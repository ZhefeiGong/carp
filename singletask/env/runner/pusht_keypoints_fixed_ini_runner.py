import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import time
import dill
import math
import wandb.sdk.data_types.video as wv
from env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from env.gym_util.async_vector_env import AsyncVectorEnv
from env.gym_util.sync_vector_env import SyncVectorEnv
from env.gym_util.multistep_wrapper import MultiStepWrapper
from env.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from env.common.pytorch_util import dict_apply
from env.runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsFixedIniRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_test=56,
            n_test_vis=56,
            test_start_seed=10000,
            legacy_test=False,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            reset_to_state=None,
            render_action=True,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        # initialize environments
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        reset_to_state=reset_to_state, # agent and block initialization
                        render_action=render_action, # whether render the action as a red cross
                        render_size=256,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # build drills for test environment
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis
            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, vae, ar, normalizer):  
        
        # model info
        device = next(ar.parameters()).device
        env = self.env

        # test
        count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.4f}M'
        duration_repo = []
        times_repo = []
        device_str = f'[#device] : {device}'
        params_str = f'[#para] : ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('vae', vae),('ar', ar),('normalizer', normalizer),)])
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        
        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        # actions
        all_actions = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            # policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False

            exe_duration = 0.0
            exe_times = 0
            exe_actions = []

            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                
                # # run policy
                # with torch.no_grad():
                #     action_dict = policy.predict_action(obs_dict)
                
                # run policy
                tmp_start_time = time.perf_counter()
                with torch.no_grad():
                    # get data
                    assert 'obs' in obs_dict
                    nobs = normalizer['obs'].normalize(obs_dict['obs'])
                    # assert nobs.shape[1]==self.n_obs_steps
                    # nobs = nobs.view(nobs.shape[0],-1)
                    # predict
                    action_pred = ar.autoregressive_infer_cfg(nobs=nobs, vae_proxy=vae) # -> B1LC
                    action_pred = action_pred.view(action_pred.shape[0],action_pred.shape[2],action_pred.shape[3])  # -> BLC
                    # unnormalize prediction
                    action_pred = normalizer['action'].unnormalize(action_pred)
                    # get action
                    start = self.n_obs_steps - 1
                    end = start + self.n_action_steps
                    action = action_pred[:,start:end]
                    action_dict = {
                        'action': action,
                        'action_pred': action_pred
                    }
                tmp_end_time = time.perf_counter()
                exe_duration += tmp_end_time - tmp_start_time
                exe_times += 1
                
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                
                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                
                # store the executable actions
                exe_actions.append(action)

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            
            duration_repo.append(exe_duration)
            times_repo.append(exe_times)
            all_actions.append(np.concatenate(exe_actions, axis=-2))

        # log
        all_actions = np.concatenate(all_actions, axis=0)
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the ariance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
                npy_path = video_path.replace('.mp4', '.npy')
                np.save(npy_path, all_actions[i]) # save trajectory here

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        # self-define params
        log_data['device_str'] = device_str
        log_data['params_str'] = params_str
        log_data['duration'] = "[#duration]: " + "; ".join(
            f"{duration:.4f}-{time}" for duration, time in zip(duration_repo, times_repo)
        )

        return log_data


