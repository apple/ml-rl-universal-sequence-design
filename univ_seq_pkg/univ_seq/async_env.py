#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
from univ_seq.upo_env import UPOEnv

######################################################################
# asynchronous bler/reward generation
######################################################################
import multiprocessing as mp
from queue import Queue

class AsyncRewardGen:

    @staticmethod
    def worker(conn, cfg, device, seed):

        process_name = mp.current_process().name
        print(f"{process_name} started")
        # override device on workers
        device = torch.device('cpu')
        env = UPOEnv(cfg, device)
        env.reset(seed=seed, sim_only=True)

        while True:
            try:
                req, data = conn.recv()
#                print(f"{process_name} req {req}, data {data}")
                if req == 'bler':
                    K, N, fbm = data
                    snr = env.get_target_snr(K,N)
                    bler = env.get_bler(fbm, N, K, snr, max_err=cfg.max_err_valid)
#                    print(f"{process_name} bler {bler}")
                    conn.send((bler, req))
                elif req == 'reward':
                    K, N, fbm, terminated = data
                    if terminated:
                        snr = env.get_target_snr(K,N)
                        bler = env.get_bler(fbm, N, K, snr, max_err=cfg.max_err_valid)
                        reward = - np.log(bler)
                    else:
                        reward = 0
#                    print(f"{process_name} reward {reward}")
                    conn.send((reward, req))
                elif req == 'reset_seq':
                    learned_seq = data
                    env.reset(learned_seq=learned_seq)
                    rcode = 0
                    conn.send((rcode, req))
                else:
                    print('Unknown request')
            except EOFError:
                print(f"{process_name} Closing connection")
                conn.close()
                break

    def __init__(self, cfg, device):
        ctx = mp.get_context('spawn')
        self.n_envs = cfg.n_envs
        self.processes = []
        self.connections = []
        for rank in range(cfg.n_envs):
            seed = cfg.seed * cfg.n_envs + rank
            parent_conn, child_conn = ctx.Pipe()
            self.connections.append(parent_conn)
            proc = ctx.Process(target=AsyncRewardGen.worker, args=(child_conn, cfg, device, seed))
            proc.start()
            self.processes.append(proc)

    def compute(self):
        pass

    def reset(self):
        pass

    def send(self, req, id):
        self.connections[id].send(req)

    def recv(self, id):
        return self.connections[id].recv()

    def close(self):
        for i in range(self.n_envs):
            self.connections[i].close()
            self.processes[i].join()


######################################################################
# custom vec env
######################################################################
from typing import Any
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import _stack_obs
import gymnasium as gym

class AsyncVecEnv(VecEnv):
    def __init__(self, cfg, device, env_dicts=None):
        print('Using AsyncVecEnv type', cfg.use_async_vec_env)
        self.cfg = cfg
        self.n_envs = cfg.n_envs
        if env_dicts is not None:
            self.envs = [UPOEnv(cfg, device, **env_dicts[i]) for i in range(self.n_envs)]
        else:
            self.envs = [UPOEnv(cfg, device) for _ in range(self.n_envs)]
        if cfg.use_async_vec_env == 1:
            self.reward_gen = AsyncRewardGen(cfg, device)
        else:
            raise RuntimeError('Unknown async vec_env type')
        self.rewards_vec = [[] for _ in range(self.n_envs)]
        self.wait_queue = Queue(maxsize = cfg.n_steps)
        super().__init__(self.n_envs, self.envs[0].observation_space, self.envs[0].action_space)

    def get_reward_gen(self):
        return self.reward_gen

    def wait_queue_empty(self):
        return self.wait_queue.empty()

    def reset(self, learned_seq=None):
        cfg = self.cfg
        seeds = [cfg.seed * cfg.n_envs + rank for rank in range(self.n_envs)]
        if learned_seq is not None:
            results = [env.reset(learned_seq=learned_seq) for env, seed in zip(self.envs, seeds)]
        else:
            results = [env.reset(seed=seed) for env, seed in zip(self.envs, seeds)]
        obs, self.reset_infos = zip(*results)
        return _stack_obs(obs, self.observation_space)

    def step_amend(self, env, action, probs, lazy): # remove probs
        observation, reward, terminated, truncated, info = env.step(action, probs=probs, lazy=lazy)
        done = terminated or truncated
        reset_info = {}
        if done:
            observation, reset_info = env.reset()
        return observation, reward, done, info, reset_info

    def step_lazy(self, actions: np.ndarray, dists=None): # remove dists
        results = [self.step_amend(env, action, dist, lazy=True) for env, action, dist in zip(self.envs, actions, dists)]
        obs, _, dones, infos, _ = zip(*results)
        for i, info in enumerate(infos):
#            print(f"step_lazy() i {i}, req {info['req_info']}")
            self.reward_gen.send(info['req_info'], i)
        self.wait_queue.put(dones)
        return _stack_obs(obs, self.observation_space), np.stack(dones), infos

    def step_reward(self):
        rewards = [self.reward_gen.recv(i) for i in range(self.n_envs)]
        rewards, tags = zip(*rewards)
        dones = self.wait_queue.get(block=False)
#        print(f'step_reward() rewards {rewards}, dones {dones}')
        # monitor rewards - add logic from Monitor class
        infos = []
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.rewards_vec[i].append(reward)
            info = {}
            if done:
                ep_rew = sum(self.rewards_vec[i])
                ep_len = len(self.rewards_vec[i])
                ep_info = {"r": round(ep_rew, 6), "l": ep_len}
                info['episode'] = ep_info
                # reset rewards
                self.rewards_vec[i] = []
            infos.append(info)
 #       for tag in tags:
 #           assert tag == 'reward'
        return np.stack(rewards), infos

    def close(self):
       self.reward_gen.close()

    def step_async(self, actions: np.ndarray) -> None:
        raise Exception('Not implemented')
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        raise Exception('Not implemented')
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment (see base class)."""
        if attr_name == "render_mode":
            return [None for _ in range(self.n_envs)]
        raise Exception('Not implemented')
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        raise Exception('Not implemented')
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        raise Exception('Not implemented')
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]


    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        raise Exception('Not implemented')
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]


