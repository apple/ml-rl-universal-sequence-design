#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
from copy import deepcopy
from stable_baselines3.common.graph import GenericGraphSpace
from univ_seq.upo_graph import gen_upo_graph
from univ_seq.cpp.polar import PolarSimulator, MultiThreadPolarSimulator
from univ_seq.polar.upo import UPO
from numpy.polynomial import polynomial as poly

# gymnasium integration
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.spaces.space import Space
#from torch_geometric.data import HeteroData
from torch_geometric.data import Data

######################################################################
# define custom graph space
######################################################################
class UPOGraphSpace(Space[Data], GenericGraphSpace):
    def __init__(self):
        super().__init__(None, None, None)

class Space:
    def __init__(self, seq):
        self.seq = seq

    def __str__(self):
        return f"{len(self.seq)} {self.seq}"

######################################################################
# approximate gap
######################################################################
x_min=16
n_seg=3
segments_table = {
    128 : [70, 112],
    256 : [160, 240],
    512 : [300, 480],
    1024 : [600, 980],
    2048 : [1200, 1980],
}
c_table={}
c_table[128] = [
    np.array([-1.8938752658012383, 1.273673912258048, -0.08547124211719509, 0.0028781642441858134, -5.1642320603640144e-05, 4.7221761816687927e-07, -1.7304260971589198e-09]),
    np.array([928.3608964919086, -61.48224975670253, 1.6971786749054116, -0.02485783246569008, 0.0002038673594236413, -8.880536929403352e-07, 1.6062509269318948e-09]),
    np.array([86496.53459062018, -3624.142520140477, 60.66411155397885, -0.5070305613116891, 0.0021158437348035362, -3.526371135696945e-06]),
]
c_table[256] = [
    np.array([5.541537437279732, -0.0006626305962462105, -0.00011484253748857537, 2.839504792085227e-06, -2.6925107271834263e-08, 1.2845488147702394e-10, -2.398072029417913e-13]),
    np.array([1638.2788846867084, -50.28926303972585, 0.6427592580929778, -0.004364966230931135, 1.6617825332719428e-05, -3.3639351449626486e-08, 2.8305209476852696e-11]),
    np.array([3135850.6317628883, -63082.29582653956, 507.4603476609728, -2.040535752851661, 0.004101371595027153, -3.296410941797844e-06]),
]
c_table[512] = [
    np.array([5.675868127900954, -0.00864352285079479, 7.497973130577298e-05, -3.0886685999660036e-07, 4.798333516510038e-10, 1.0223024895221107e-12, -2.948431919429317e-15]),
    np.array([-17.327992802721297, -0.028858382293282105, 0.002853642656366239, -1.9734201259373416e-05, 5.887017771970213e-08, -8.35278717991562e-11, 4.63879285725806e-14]),
    np.array([-7833378.623879854, 79530.47234393448, -322.96461620477544, 0.6557294454302263, -0.0006656470498405703, 2.7027409071931447e-07]),
]
c_table[1024] = [
    np.array([5.7229360646616945, -0.00852897183449237, 5.340244882615343e-05, -1.9311557720057298e-07, 4.219862659567054e-10, -4.895597841870437e-13, 2.3343873518926586e-16]),
    np.array([328.6777219012107, -2.6635309564401313, 0.009119577509045218, -1.662355514174505e-05, 1.7024922402164385e-08, -9.287397067782773e-12, 2.1101673298099096e-15]),
    np.array([-84152059.66101196, 421370.5888096302, -843.9451348158916, 0.8451330190595114, -0.0004231526859033633, 8.474628145210905e-08]),
]
c_table[2048] = [
    np.array([5.700281546838879, -0.0062718083999095935, 2.278340316005537e-05, -4.4933294831008464e-08, 5.011476191747211e-11, -2.876399534171889e-14, 6.639678318733661e-18]),
    np.array([282.64624303886427, -1.183674187089663, 0.00209203285707641, -1.9621694366998937e-06, 1.0307741188425581e-09, -2.875493361075763e-13, 3.331313525215739e-17]),
    np.array([-356960836.1011043, 889292.8396641049, -886.1785925652216, 0.44152987368082464, -0.00010999208731951715, 1.096012323953536e-08]),
]

######################################################################
# helpers
######################################################################
def print_dist(probs):
    n_actions = probs.size
    actions = np.arange(n_actions)
    indices = probs != 0.0
    probs = probs[indices]
    actions = actions[indices]
    field = 'action'
    print(f'{field:>8}: ' , end='')
    for i in range(actions.size):
        print(f'{actions[i]:>5} ', end='')
    print('')
    field = 'prob'
    print(f'{field:>8}: ' , end='')
    for i in range(probs.size):
        print(f'{probs[i]:.3f} ', end='')
    print('')

######################################################################
# define environment
class UPOEnv(gym.Env):
    def __init__(self, cfg, device, cnt=None):

        self.cfg = cfg
        #self.graph_arch = cfg.graph_arch
        #self.N = cfg.N
        #self.K = cfg.K
        self.N = None
        self.K = None
        self.N_max = cfg.N_max
        self.N_set = cfg.N_set
        self.K_range = cfg.K_range
        self.L = cfg.L
        self.crc_poly = np.array(cfg.crc_poly)
        self.conv_poly = np.array(cfg.conv_poly)
        self.genie_aided = cfg.genie_aided
        self.num_runs = cfg.num_runs
        self.num_threads = cfg.num_threads
        self.snr_range = cfg.train_snr_range
#        self.snr_offset = cfg.train_snr_offset
        self.snr_offsets = cfg.train_snr_offsets
        self.approx_capacity_gap = cfg.approx_capacity_gap
        self.device = device
        #self.state = None # graph
        self.states = {} # graph
        self.SNRdB = None
        self.fbm   = None
        # reward specific values
        self.log_bler = None
        self.max_err = cfg.max_err
        self.max_runs = cfg.max_runs
        self.min_bler = 1/cfg.max_runs
        # GAT experiment
        self.add_self_loops = cfg.add_self_loops
        # range expansion
        self.range_expansion = cfg.range_expansion
        self.num_episodes = cfg.num_episodes
        self.cnt = 0 if cnt is None else cnt
        # iterative learning
        self.iterative_learning = cfg.iterative_learning
        self.lookahead_window = cfg.lookahead_window
        # multi-stage training
        self.seqs = {}
        self.multi_stage_training = cfg.multi_stage_training
#        print('Iterative learning active') if self.iterative_learning == True else None
        if self.multi_stage_training or (self.iterative_learning and cfg.learned_seq is not None):
#            print(f'learned_seq {Space(cfg.learned_seq)}')
            seq = cfg.learned_seq
            for N in cfg.N_set:
                self.seqs[N] = seq[seq < N]
        # save intermediate state
        self.init_upos = {}

        # UPO
        self.upos = {}
        self.num_past_actions = cfg.num_past_actions
        self.num_hops = cfg.num_hops

        # simulator setup (set seed)
        self.simulator = None
        if not cfg.use_sb3:
            self.reset_simulator(cfg.seed)

        # setup UPO instances
        for N in cfg.N_set:
            lower_N = N//2
            lower_n_seq = cfg.learned_seq_table[lower_N]['learned_seq'].squeeze() if cfg.embed_lower_n else None
            cfg.K_min_embed = cfg.K_min_embed if cfg.K_min_embed is not None else cfg.K_min
            self.upos[N] = UPO(N, reverse=cfg.reverse, lower_n_seq=lower_n_seq, promote_ln_node=cfg.promote_ln_node, K_min=cfg.K_min_embed)

        # monitor multi-N violations
        self.monitor_lower_n = cfg.monitor_lower_n
        self.learned_seq_table = cfg.learned_seq_table

        print('Monitoring lower N violations') if self.monitor_lower_n == True else None
#        print('Using Approximate capacity gap') if self.approx_capacity_gap == True else None

        # Observation space
        self.observation_space = UPOGraphSpace() # dummy space
        # Action space
        self.action_space = gym.spaces.Discrete(cfg.N_max) # max N

    def get_state(self):
        return { 'cnt' : self.cnt }

    def reset_simulator(self, seed):
        seed &= 0xffff # mask off LSBs to avoid overflowing int32
#       self.simulator = PolarSimulator(seed)
        self.simulator = MultiThreadPolarSimulator(seed, self.num_runs, self.num_threads)

    def get_capacity_gap(self,K,N):
        assert self.genie_aided == True
        assert K >= x_min
        index = 0
        for i in range(n_seg-1):
            if K > segments_table[N][i]:
                index = i+1
#        print('K {K} N {N} segments {segments_table[N]} index {index}')
        c = c_table[N][index]
        gap = poly.polyval(K, c)
        return gap

    def get_target_snr(self,K,N):
        # compute target SNR
        # FIXME: add param E for rate-matched targets
        R = K/N
        capacityLimitdB = 10*np.log10(2**R - 1)

        if self.approx_capacity_gap:
            gap = self.get_capacity_gap(K,N)
        else:
            gap = self.snr_offsets[N]

        SNRdB = capacityLimitdB + gap

        return SNRdB

    def get_graph(self, fbm, node_indices, actions, N, K, SNRdB, reset=False):
        # just call directly for now
        return gen_upo_graph(self.N_max, N, K, SNRdB, fbm, node_indices, actions).to(self.device)

    def reset(self, seed = None, get_seq = False, sim_only = False, learned_seq = None):

        cfg = self.cfg

        if learned_seq is not None:
            print(f"Resetting learned seq")
            seq = learned_seq
            for N in cfg.N_set:
                self.seqs[N] = seq[seq < N]
            # reset cache
            self.init_upos = {}

        # reset simulator
        if seed is not None:
#            print(f"Env.reset(): seed {seed}")
            self.reset_simulator(seed)

        if get_seq:
            N = np.max(self.N_set)
            K = N
        else:
            # sample code dimension and code length
            # snr must be set at the transition region to ease learning
            N = np.random.choice(self.N_set)
            K_min = cfg.K_min
            K_end = cfg.K_max_dict[N]
            if self.multi_stage_training:
                K_min = cfg.K_start + 1
                K_end = cfg.K_end
            K = np.random.randint(K_min, K_end+1)
            if self.range_expansion:
                n_steps = cfg.range_expansion_steps
                K_intv = K_end - K_min
                K_end = K_min + K_intv//n_steps
                for step in range(1,n_steps):
                    num_episodes_to_full = cfg.num_episodes * cfg.range_expansion_end
                    if self.cnt >= (num_episodes_to_full * step // n_steps):
                        K_end = K_min + K_intv * (step + 1) // n_steps
                    else:
                        break
                K = np.random.randint(K_min, K_end+1)
            if self.iterative_learning:
                if N in self.seqs:
                    seq = self.seqs[N]
                    K_min = seq.size + 1
                K_end = K_min + self.lookahead_window - 1
                K_end = min(K_end, N) # upper bound at N
                K = np.random.randint(K_min, K_end+1)
            # check snr offset
            snr_offset = self.snr_offsets[N]
            if self.approx_capacity_gap:
                snr_offset = self.get_capacity_gap(K, N)
            print(f'K_range/N [{K_min},{K_end}]/{N}, K {K}, snr_offset {snr_offset:.2f}')

        # compute target SNR
        SNRdB = self.get_target_snr(K,N)
        # set fbm depending on N
        fbm = np.zeros((N,), dtype=np.integer) # all frozen
        if self.multi_stage_training or (self.iterative_learning and N in self.seqs):
            # initialize pre-assigned info bits
            seq_N = self.seqs[N]
            K_start = cfg.K_start if self.multi_stage_training else seq_N.size
            indices = seq_N[-K_start:]
            fbm[indices] = 1

        #print(f'reset: K/N {K}/{N} snr {SNRdB:.2f}')
        self.upos[N].reset()
        if self.multi_stage_training or (self.iterative_learning and N in self.seqs):
            if not N in self.init_upos:
                self.upos[N].fast_forward(self.seqs[N], K_start, num_hops=self.num_hops)
                self.init_upos[N] = deepcopy(self.upos[N])
            else:
                # restore state
                self.upos[N] = deepcopy(self.init_upos[N])
        actions, scope = self.upos[N].get_actions_and_scope(num_hops=self.num_hops, num_past_actions=self.num_past_actions)

        graph = self.get_graph(fbm, scope, actions, N, K, SNRdB, reset=True) if not sim_only else None

        self.states[N] = graph
        self.SNRdB = SNRdB
        self.fbm   = fbm
        self.N     = N
        self.K     = K
        self.cnt  += 1

        return graph, {}


    def get_action_space(self):
        return self.upos[self.N].get_actions(num_hops=self.num_hops)

    def action_space_sample(self):
        # take random action
        action = np.random.choice(self.upos[self.N].get_actions())
        return action

    def get_bler(self, fbm, N, K, SNRdB, max_err):

        crc_coeffs = np.array([])
        frozen_bits = 1 - fbm
        bler = self.simulator.get_bler(N, self.L, SNRdB, max_err, self.max_runs, frozen_bits, self.crc_poly, self.genie_aided, crc_coeffs)

        if bler == 0:
            bler = self.min_bler
            expected = self.genie_aided and (K <= np.log2(self.L).astype(int))
            if not expected:
                print(f'Warning: K/N {K}/{N} snr {SNRdB:.2f} bler 0')

        return bler

    def reward(self, fbm):

        bler = self.get_bler(fbm, self.N, self.K, self.SNRdB, self.max_err)
        log_bler = np.log(bler)
        reward = - log_bler

        return reward


    def step(self, action, probs=None, no_reward = False, lazy = False): # remove probs

        old_fbm = self.fbm
        fbm = self.fbm.copy()

        assert fbm[action] == 0
        # set bit to info
        fbm[action] = 1

        # check for lower N order violations
        eval_mode = no_reward
        if self.monitor_lower_n and not eval_mode:
            N = self.N
            lengths = np.array([128, 256, 512, 1024])
            lower_lengths = lengths[lengths < N]
            for lower_N in lower_lengths:
                data = self.learned_seq_table[lower_N]
                learned_seq_N = data['learned_seq'].squeeze()
                prob_dists = data['prob_dists'].squeeze()
                indices = np.where(learned_seq_N == action)[0]
                past_actions = self.upos[self.N].used_nodes
                if indices.size > 0 and indices[0] < (lower_N - 1):
                    index = indices[0]
                    higher_order_action = learned_seq_N[index + 1]
                    if not higher_order_action in past_actions:
                        k_th = past_actions.size + 1
                        print(f'*** {k_th}-th action violated order in lower N {lower_N} ***')
                        print(f'*** taking action {action} before {higher_order_action} ***')
                        print(f'*** policy distribution at {higher_order_action} ***')
                        probs_N = prob_dists[index + 1]
                        print_dist(probs_N)
                        print(f'*** current policy distribution ***')
                        print_dist(probs)
                        if index < (lower_N - self.cfg.K_min):
                            print('*** penalizing action [not implemented] ***')

        self.upos[self.N].update_action(action)
        actions, scope = self.upos[self.N].get_actions_and_scope(num_hops=self.num_hops, num_past_actions=self.num_past_actions, eval_mode=eval_mode)
        next_state = self.get_graph(fbm, scope, actions, self.N, self.K, self.SNRdB)

        reward = 0
        terminated = False
        truncated = False
        # reward at the end after K allocations
        if next_state.num_info_unassigned == 0:
            reward = self.reward(fbm) if not (no_reward or lazy) else 0
            terminated = True

        #print(f'state {old_fbm} action {action} reward {reward:.3f} nstate {fbm}')

        # save next state
        self.states[self.N] = next_state
        self.fbm   = fbm

        info = {}
        # add req_info in lazy mode
        if lazy:
            fbm = fbm if terminated else None
            req_info = ('reward', (self.K, self.N, fbm, terminated))
            info['req_info'] = req_info

        return next_state, reward, terminated, truncated, info

