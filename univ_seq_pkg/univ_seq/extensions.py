#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import time
import warnings
import numpy as np
import gymnasium as gym
import torch
import torch.distributed as dist
from scipy.io import savemat
from utils import PriorityQueue

################################################################################
# debug/helper functions
################################################################################
def dump_probs(model, env, cfg):
    # dump probs
    probs_list = []
    actions_list = []
    N = cfg.N_set[-1]
    K_max = cfg.K_max_dict[N]
    print(f'Dump config: N {N} K_max {K_max}')
    obs, _ = env.reset(get_seq=True)
    for _ in range(K_max):
        dist = model.policy.get_distribution(obs)
        probs_list.append(dist.probs.squeeze())
        action, state = model.predict(obs, deterministic=cfg.deterministic_eval)
        action = action.detach().cpu().numpy()
        actions_list.append(action)
        obs, reward, term, trunc, _ = env.step(action)
    probs_mat = torch.stack(probs_list).detach().cpu().numpy()
    actions = np.array(actions_list)
    return actions, probs_mat

class Space:
    def __init__(self, seq):
        self.seq = seq

    def __str__(self):
        return f"{len(self.seq)} {self.seq}"

################################################################################
# vec_env setup
################################################################################
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from univ_seq.upo_env import UPOEnv
from univ_seq.async_env import AsyncVecEnv, AsyncRewardGen
def vec_env_setup(model, cfg, device, reset=True):

    env_dicts = None
    # clear env_dicts if reset=True
    if model and hasattr(model, 'env_dicts') and not reset:
        env_dicts = model.env_dicts

    if cfg.iterative_learning:
        if hasattr(model, 'learned_seq') and model.learned_seq.size > 0:
            print('detected previously learned seq')
            print('seq', model.learned_seq)
            cfg.learned_seq = model.learned_seq

    if cfg.multi_stage_training:
        cfg.learned_seq = model.best_seq

    if cfg.use_subproc_vec_env:
        print(f'Using multiprocessing vec env')
        vec_env_type = SubprocVecEnv
    else:
        vec_env_type = DummyVecEnv

    if cfg.use_async_vec_env:
        vec_env = AsyncVecEnv(cfg, device, env_dicts)
    else:
        env_kwargs = dict(cfg=cfg, device=device)
        if env_dicts:
            for env_dict in env_dicts:
                env_dict.update(env_kwargs)
        vec_env = make_vec_env(UPOEnv, n_envs=cfg.n_envs, env_kwargs=env_kwargs, vec_env_cls=vec_env_type, env_dicts=env_dicts)
    return vec_env

def get_bler_helper(K_set, N, seq, cfg, env, reward_gen):
    # only for one N
    bler_vec = []
    num_bler = np.array(K_set).size
    if cfg.use_async_vec_env:
        reward_gen.reset()
        for i in range(num_bler):
            K = K_set[i]
            fbm = np.zeros((N,), dtype=int)
            seq_N = seq[ seq < N ]
            idx = seq_N[-K:] # indices of last K elements
            fbm[idx] = 1
            req = 'bler', (K, N, fbm)
            reward_gen.send(req, i)
        reward_gen.compute()
        for i in range(num_bler):
            K = K_set[i]
            snr = env.get_target_snr(K,N)
#           bler = reward_gen.recv(i)
            bler, tag = reward_gen.recv(i)
            assert tag == 'bler'
            bler_vec.append(bler)
            print(f'K/N {K}/{N} snr {snr:.2f} bler {bler:.5f}')
    else:
        for i in range(num_bler):
            K = K_set[i]
            fbm = np.zeros((N,), dtype=int)
            seq_N = seq[ seq < N ]
            idx = seq_N[-K:] # indices of last K elements
            fbm[idx] = 1
            snr = env.get_target_snr(K,N)
            bler = env.get_bler(fbm, N, K, snr, max_err=cfg.max_err_valid)
            bler_vec.append(bler)
#           print(f'K/N {K}/{N} snr {snr:.2f} bler {bler:.5f} fbm {fbm}')
            print(f'K/N {K}/{N} snr {snr:.2f} bler {bler:.5f}')
    return bler_vec

################################################################################
# Eval callback
# incorporate current validation strategy (strip out most things)
################################################################################
def evaluate_policy(model, env, reward_gen, cfg):

    best_avg_bler = 1.0
    best_seq = None
    best_blers = None
    complete = False

    def get_length(cfg, learned_seq, N, init_step=False):
        curr_len = learned_seq.size
        rem_len = N - curr_len
        if init_step:
            length = cfg.K_min
        elif rem_len < cfg.freeze_set:
            length = rem_len
        else:
            length = cfg.freeze_set
        return length

    for i_epi in range(cfg.n_eval_episodes):
        print(f'episode {i_epi}')
        # deterministic for 1st episode
        deterministic_eval = True if i_epi == 0 else False
        # get sequence
        seq_len = np.max(cfg.N_set)
        seq = np.full((seq_len,), -1, dtype=int)

        K_start = 0 # default
        if cfg.multi_stage_training:
            K_start = cfg.K_start
            seq[-K_start:] = cfg.learned_seq[-K_start:]
    #            print('learned_seq', cfg.learned_seq)
    #            print('seq', seq)
        if cfg.iterative_learning:
            partial_seq = model.learned_seq
            print('learned seq', Space(partial_seq))
            K_start = partial_seq.size
            init_step = partial_seq.size == 0
            if partial_seq.size > 0:
                seq[-K_start:] = cfg.learned_seq
            if deterministic_eval:
                length = get_length(cfg, model.learned_seq, seq_len, init_step)
                init_actions = np.zeros((length,), dtype=int)
                init_entropy_v = np.zeros((length,))
                prob_dists = np.zeros((length, seq_len))
        obs, _ = env.reset(get_seq=True)
        for i, k in enumerate(range(K_start, seq_len)):
            action, state = model.predict(obs, deterministic=deterministic_eval)
            dist = model.policy.get_distribution(obs)
            values, log_prob, entropy = model.policy.evaluate_actions(obs, action)
            action = action.item() # convert it to python scalar
            entropy = entropy.item()
            if cfg.iterative_learning and deterministic_eval:
                length = get_length(cfg, model.learned_seq, seq_len, init_step)
                if i < length:
                    init_actions[-1-i] = action
                    init_entropy_v[-1-i] = entropy
                    prob_dists[-1-i] = dist.probs.squeeze().detach().cpu().numpy()
            seq[-1-k] = action
            next_obs, reward, terminated, truncated, _ = env.step(action, no_reward=True)
            done = terminated or truncated
            obs = next_obs
            if done:
                break

        print(f'seq {seq}')
        bler_vec = []
        if cfg.iterative_learning:
            assert len(cfg.N_set) == 1
            # focus on the freeze set only
            # this is myopic (somewhat greedy)
            N = cfg.N_set[0]
            length = get_length(cfg, model.learned_seq, N)
            if init_step:
                K_set = cfg.K_min + np.arange(length)
            else:
                K_set = K_start + 1 + np.arange(length)
            bler_vec_N = get_bler_helper(K_set, N, seq, cfg, env, reward_gen)
            bler_vec.extend(bler_vec_N)
        else:
            # evaluate all N's and K's
            for N in cfg.N_set:
                K_set = cfg.K_valid_dict[N]
                bler_vec_N = get_bler_helper(K_set, N, seq, cfg, env, reward_gen)
                bler_vec.extend(bler_vec_N)

        bler_vec = np.array(bler_vec)
        avg_bler = bler_vec.mean()
        if avg_bler < best_avg_bler:
            best_avg_bler = avg_bler
            best_seq = seq
            best_blers = bler_vec

    mat_dict = {}
    num_matched = 0
    if cfg.iterative_learning:
        if init_step:
            # compare set equivalence instead of exact sequence match
            last_actions_sorted = np.sort(np.array(model.last_actions))
            init_actions_sorted = np.sort(init_actions)
            mean_entropy = init_entropy_v.mean()
            print(f'last_actions\n{last_actions_sorted}')
            print(f'init_actions\n{init_actions_sorted}')
            print(f'last_entropy {model.last_entropy_v}, curr_entropy {mean_entropy}')
            print(f'episode_num {model._episode_num}, max_init_train {cfg.min_initial_training}')
            if len(model.last_actions) > 0:
                num_matched = np.sum(np.all(last_actions_sorted == init_actions_sorted, axis=-1))
            if (len(model.last_actions) >= cfg.settle_threshold and np.all(last_actions_sorted == init_actions_sorted) and
                model._episode_num > cfg.min_initial_training):
                model.learned_seq = init_actions
                model.prob_dists = prob_dists
                model.last_actions.clear()
                model.last_entropy_v.clear()
                cfg.learned_seq = model.learned_seq
                # save dict
                mat_dict = { 'learned_seq': model.learned_seq,
                             'prob_dists': model.prob_dists,
                             'K_set': K_set
                            }
                print('Set new vec_env')
                print('learned_seq', cfg.learned_seq)
                model.updated_vec_env = True
            else:
                model.last_actions.append(init_actions)
                model.last_entropy_v.append(mean_entropy)
        else:
            # exact sequence match required
            mean_entropy = init_entropy_v.mean()
            print(f'last_actions {model.last_actions}, init_actions {init_actions}')
            print(f'last_entropy {model.last_entropy_v}, curr_entropy {mean_entropy}')
            if len(model.last_actions) > 0:
                num_matched = np.sum(np.all(np.array(model.last_actions) == init_actions, axis=-1))
            # end eval if mean_entropy is zero, i.e. only 1 action per state
            if len(model.last_actions) >= cfg.settle_threshold and np.all(np.array(model.last_actions) == init_actions) or mean_entropy == 0.0:
                model.learned_seq = np.concatenate((init_actions, model.learned_seq))
                model.prob_dists = np.concatenate((prob_dists, model.prob_dists))
                model.last_actions.clear()
                model.last_entropy_v.clear()
                cfg.learned_seq = model.learned_seq
                if model.learned_seq.size == cfg.N_set[0]:
                    complete = True
                # save dict
                mat_dict = { 'learned_seq': model.learned_seq,
                             'prob_dists': model.prob_dists,
                             'K_set': K_set,
                            }
                if not complete:
                    print('Set new vec_env')
                    print('learned_seq', cfg.learned_seq)
                    model.updated_vec_env = True
            else:
                model.last_actions.append(init_actions)
                model.last_entropy_v.append(mean_entropy)

    # report progress
    prog_dict = {}
    if cfg.iterative_learning:
        prog_dict = { 'curr_K': np.min(K_set),
                      'match_length': num_matched,
                      'mean_entropy': mean_entropy,
                     }

    return best_avg_bler, best_seq, best_blers, mat_dict, prog_dict, complete

################################################################################
# Eval callback
# incorporate current validation strategy (strip out most things)
################################################################################
from typing import Optional, Union, Any
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        reward_gen,
        cfg,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        log_path: Optional[str] = None,
        data_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.cfg = cfg
        self.candidates = PriorityQueue(cfg.num_candidates)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = cfg.n_eval_episodes
        self.eval_freq = cfg.eval_freq
        #self.best_mean_reward = -np.inf
        #self.last_mean_reward = -np.inf
        self.best_avg_bler = 1.0
        self.last_avg_bler = 1.0
        self.deterministic = cfg.deterministic_eval
        self.warn = warn

        # Convert to VecEnv for consistency
        #if not isinstance(eval_env, VecEnv):
        #    eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.reward_gen = reward_gen
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.data_path = data_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
#        if not isinstance(self.training_env, type(self.eval_env)):
#            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        """
        Evaluation step
        """
        continue_training = True

        if self.cfg.use_ddp:
            # only master process does evaluation
            master_process = dist.get_rank() == 0
            if not master_process:
                return continue_training

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            eval_start = time.perf_counter()
            avg_bler, seq, blers, mat_dict, prog_dict, complete = evaluate_policy(
                self.model,
                self.eval_env,
                self.reward_gen,
                self.cfg,
            )
            self.eval_time = time.perf_counter() - eval_start

            # update eval_env
            if self.model.updated_vec_env and not complete:
                self.eval_env.reset(learned_seq=self.model.learned_seq)

            if self.cfg.iterative_learning and 'learned_seq' in mat_dict:
                learned_seq_path = os.path.join(self.data_path, self.cfg.identity + "_learned_seq.mat")
                print('Save learned seq to file', learned_seq_path)
                savemat(learned_seq_path, mat_dict)
                best_model_path = os.path.join(self.best_model_save_path, self.cfg.identity + "_best_model.zip")
                print('Saving model to file', best_model_path)
                self.model.save(best_model_path)

            if avg_bler < self.best_avg_bler:
                if self.verbose >= 1:
                    print("New best avg bler!")
                if self.best_model_save_path is not None:
                    # save best sequence
                    self.model.best_seq = seq
                    best_model_path = os.path.join(self.best_model_save_path, self.cfg.identity + "_best_model.zip")
                    print('Saving best model to file', best_model_path)
                    # FIXME: save handling for DDP model?
                    self.model.save(best_model_path)
                self.best_avg_bler = avg_bler

                # save best N candidates
                save_list = self.candidates.add((blers, seq))

                if save_list and not self.cfg.iterative_learning:
                    # save candidates list
                    blers, seqs = list(zip(*self.candidates))
                    seqs = np.array(seqs)
                    blers = np.array(blers)
                    mat_dict = {"rate_profiles": seqs,
                                "blers"        : blers,
                                }

                    rate_profiles_path = os.path.join(self.data_path, self.cfg.identity + "_rate_profiles.mat")
                    print('Save candidates to file', rate_profiles_path)
                    savemat(rate_profiles_path, mat_dict)

            # Add to current Logger
            self.logger.record("eval/avg_bler", avg_bler)
            self.logger.record("eval/best_avg_bler", self.best_avg_bler)
            self.logger.record("monitor/eval_time", self.eval_time)
            if self.cfg.iterative_learning:
                self.logger.record("eval/curr_K", prog_dict['curr_K'])
                self.logger.record("eval/match_length", prog_dict['match_length'])
                self.logger.record("eval/mean_entropy", prog_dict['mean_entropy'])

            # Dump log so the evaluation results are printed with the correct timestep
            #self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            #self.logger.dump(self.num_timesteps)

            # Terminate training
            if self.cfg.iterative_learning and complete:
                continue_training = False

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


################################################################################
# schedulers
################################################################################
from typing import Any, Callable, Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func
