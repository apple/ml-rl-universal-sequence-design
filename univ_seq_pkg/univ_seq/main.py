#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import torch
import gymnasium as gym
import numpy as np
import os
import scipy.io as sio

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from collections import deque

from stable_baselines3 import PPO, SPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.logger import configure, register_output_format
from stable_baselines3.common.buffers import RolloutBuffer

import random
from univ_seq.utils import set_seed
from univ_seq.extensions import EvalCallback
from univ_seq.extensions import linear_schedule, dump_probs, vec_env_setup
from univ_seq.upo_env import UPOEnv
from univ_seq.async_env import AsyncVecEnv, AsyncRewardGen
from univ_seq.upo_gnn import GnnActorCriticPolicy
from univ_seq.get_args import get_args

if __name__ == "__main__":

    cfg = get_args()

    set_seed(cfg.seed)

    identity = str(random.random())[2:8]
    log_prefix = "./logs"
    model_prefix = "./models"
    checkpoint_prefix = "./models"
    data_prefix = "./data"
    cfg.identity = identity
    name_prefix = identity + "_model"
    model_path = identity + "_model.zip"
    model_path = os.path.join(model_prefix, model_path)
    checkpoint_path = identity + "_model_ckp.zip"
    checkpoint_path = os.path.join(checkpoint_prefix, checkpoint_path)
    file_name = os.path.basename(cfg.model_path)
    debug_path = file_name[:10] + "_debug.mat"
    debug_path = os.path.join(data_prefix, debug_path)

    print('[ID]', identity)
    print('Pytorch', torch.__version__)

    use_cuda = not cfg.use_cpu and torch.cuda.is_available()
    use_mps  = not cfg.use_cpu and torch.backends.mps.is_available()
    device = torch.device(
        "cuda" if use_cuda else
        "mps"  if use_mps else
        "cpu"
    )

    #################################################
    # DDP setup
    #################################################
    if cfg.use_ddp and (int(os.environ.get('RANK', -1)) != -1):
#       default NCCL timeout is 10 mins
#       constexpr auto kProcessGroupNCCLDefaultTimeout =
#           std::chrono::milliseconds(10 * 60 * 1000);
        nccl_timeout = timedelta(minutes=100)
        dist.init_process_group(backend='nccl', timeout=nccl_timeout)
#        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        assert dist.get_rank() == ddp_rank
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        torch.cuda.set_device(ddp_local_rank)
        device = ddp_local_rank
        seed = cfg.seed * ddp_world_size + ddp_rank
        print(f"Start running DDP on rank {ddp_rank} world size {ddp_world_size}")
        print(f"Using cuda:{device} device.")
        # Effective time step, sizes, etc
        cfg.num_timesteps = cfg.num_timesteps // ddp_world_size
#        cfg.batch_size = cfg.batch_size // ddp_world_size
#        assert cfg.batch_size >= 32, "Somethings not right..."
    else:
        ddp_world_size = 1
        master_process = True
        cfg.use_ddp = False
        seed = cfg.seed
        print(f"Using {device} device.")

    # reset seed
    print(f'Setting seed {seed}')
    set_seed(seed)

    #################################################
    # set up logging
    #################################################
    formatter_kwargs = dict(log_interval=1)

    save_path = os.path.join(log_prefix, identity + "_ppo_univ_seq")

    logger = configure(save_path, ["stdout", "csv", "tensorboard"])

    #################################################
    # set up algorithm
    #################################################
    if cfg.policy_algorithm == 'ppo':
        policy_opt_algo = PPO
        print('Running PPO')
    elif cfg.policy_algorithm == 'spo':
        policy_opt_algo = SPO
        print('Running SPO')
    else:
        raise Exception('Unknown policy algo')

    model = None
    #################################################
    # restoring from checkpoint if one exists
    # NOTE: this takes precedence over load_model
    #################################################
    if cfg.disable_resume:
        checkpoint_restored = False
        print('Ignoring checkpoints')
    elif os.path.isfile(checkpoint_path):
        # work around for circular dependency...
        # 1) load model first to reset cfg.learned_seq to latest in model
        # 2) reload model again with a properly configured vec_env
        checkpoint_restored = True
        print('Restoring from checkpoint path', checkpoint_path)
        model = policy_opt_algo.load(checkpoint_path, device=device)
        vec_env = vec_env_setup(model, cfg, device, reset=False)
        model = policy_opt_algo.load(checkpoint_path, device=device, env=vec_env)
    else:
        checkpoint_restored = False
        print('No checkpoint found')

    #################################################
    # load model from file
    #################################################
    if model is None and cfg.load_model:
        if cfg.iterative_learning and cfg.resume_iterative_learning:
            print(f'loading learned seq at {cfg.sequence_path}')
            cfg.learned_seq = sio.loadmat(cfg.sequence_path)['learned_seq'].squeeze()
        vec_env = vec_env_setup(model, cfg, device, reset=True)
        print('Loading model from file:', cfg.model_path)
        model = policy_opt_algo.load(cfg.model_path, device=device, env=vec_env)

    #################################################
    # set up environments
    #################################################
    print(f'Using {cfg.n_envs} Environments')
    # Account for the number of parallel environments
    cfg.num_episodes = cfg.num_episodes // (cfg.n_envs * ddp_world_size)

    if model is None: # needed only for fresh start
        vec_env = vec_env_setup(model, cfg, device)

    ###########################################################
    # Fixup for loaded model that does not match the current config
    ###########################################################
    if model:
        print('checking for config changes from loaded model')
        # fixup for multi-n
        if cfg.multi_n_training:
            gnn = model.policy.model
            if isinstance(gnn, DDP):
                gnn = gnn.module
            gnn.post_proc.multi_length = True
        # fixup for async_vec_env
        model.use_async_vec_env = cfg.use_async_vec_env
        # fix training parameters that may be different from new config
        if cfg.lr != model.learning_rate:
            print('detected learning rate change')
            # update learning rate / schedule
            model.learning_rate = cfg.lr
            model._setup_lr_schedule()
        if (cfg.n_steps != model.n_steps or
            cfg.gamma != model.gamma or
            cfg.gae_lambda != model.gae_lambda or
            cfg.n_envs != model.n_envs):
            print('detected rollout related changes')
            # update model params
            model.n_steps = cfg.n_steps
            model.gamma = cfg.gamma
            model.gae_lambda = cfg.gae_lambda
            model.n_envs = cfg.n_envs
            # update rollout_buffer
            model.rollout_buffer = RolloutBuffer(
                    model.n_steps,
                    model.observation_space,
                    model.action_space,
                    device=device,
                    gamma=model.gamma,
                    gae_lambda=model.gae_lambda,
                    n_envs=model.n_envs)

    #################################################
    # set up checkpointing and agent evaluation
    #################################################
    callbacks = []
    if cfg.show_progress:
        callbacks.append(ProgressBarCallback())

    if cfg.save_freq > 0:
        # Account for the number of parallel environments
        cfg.save_freq = max(cfg.save_freq // (cfg.n_envs * ddp_world_size), 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=cfg.save_freq,
                save_path=checkpoint_prefix,
                name_prefix=identity + "_model",
                checkpoint_path=checkpoint_path,
                use_ddp = cfg.use_ddp,
                verbose=2,
            )
        )

    if cfg.eval_freq > 0:
        assert cfg.eval_freq % (cfg.n_envs * cfg.n_steps) == 0
        # Account for the number of parallel environments
        cfg.eval_freq = max(cfg.eval_freq // (cfg.n_envs * ddp_world_size), 1)

        eval_env = UPOEnv(cfg, device)
        eval_env.reset(cfg.seed) # initialize env
        if cfg.use_async_vec_env == 1:
            reward_gen = AsyncRewardGen(cfg, device)
        else:
            reward_gen = None

        eval_callback = EvalCallback(
            eval_env,
            reward_gen,
            cfg,
            best_model_save_path=model_prefix,
            log_path=None,
            data_path=data_prefix,
        )
        callbacks.append(eval_callback)


    #################################################
    # set up policy network and RL algo
    #################################################
    if not (cfg.load_model or checkpoint_restored):
        policy_kwargs = dict(cfg=cfg)
        model = policy_opt_algo(GnnActorCriticPolicy,
                                vec_env,
                                policy_kwargs = policy_kwargs,
                                graph_obs_space = True,
                                use_ddp = cfg.use_ddp,
                                use_async_vec_env = cfg.use_async_vec_env,
                                batch_size = cfg.batch_size,
                                #clip_range = linear_schedule(cfg.clip_range),
                                clip_range = cfg.clip_range,
                                gamma = cfg.gamma,
                                gae_lambda = cfg.gae_lambda,
                                ent_coef = cfg.ent_coef,
                                vf_coef = cfg.vf_coef,
                                max_grad_norm = cfg.max_grad_norm,
                                #learning_rate = linear_schedule(cfg.lr),
                                learning_rate = cfg.lr,
                                n_epochs = cfg.n_epochs,
                                n_steps = cfg.n_steps,
                                device = device,
                                tensorboard_log = log_prefix,
                                verbose=1)

    model.set_logger(logger)

    # iterative learning
    model.updated_vec_env = False
    model.last_actions = deque([], maxlen=cfg.settle_threshold)
    model.last_entropy_v = deque([], maxlen=cfg.settle_threshold)
    if not hasattr(model, 'learned_seq'):
        model.learned_seq = np.array([], dtype=int)

    #################################################
    # start training
    #################################################
    if not cfg.query:
        model.learn(total_timesteps=cfg.num_timesteps, callback=callbacks, reset_num_timesteps=not checkpoint_restored)

    #################################################
    # post processing
    #################################################
    if master_process and not cfg.query:
        print('Save model to file', model_path)
        model.save(model_path)

    if cfg.query:
        actions, probs_mat = dump_probs(model, eval_env, cfg)
        mat_dict = {"actions": actions,
                    "probs_mat": probs_mat }
        print('Save debug to file', debug_path)
        sio.savemat(debug_path, mat_dict)

    if cfg.use_ddp:
        dist.destroy_process_group()

    if cfg.use_async_vec_env:
        # shutdown subprocesses
        vec_env.close()
        reward_gen.close()

