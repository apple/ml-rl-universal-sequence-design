#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from jsonargparse import ArgumentParser
from jsonargparse import ActionYesNo
from jsonargparse.typing import Path_fr
import numpy as np
import scipy.io as sio

def get_args():

    parser = ArgumentParser(prog="main", description="PCCMP GNN trainer")

    ################################################################################
    # Code/Decoder specification
    ################################################################################
    parser.add_argument("--N", type=int, default=64, help="code length")
    parser.add_argument("--K", type=int, default=32, help="information length including CRC")
    parser.add_argument("--L", type=int, default=8, help="list size")
    parser.add_argument("--crc_poly", type=list[int], default=[1,0,0,1,1], help="crc polynomial")
    parser.add_argument("--genie_aided", type=bool, default=False, help="Genie-aided SCL")
    parser.add_argument("--target_snr", type=float, default=2.0, help="design SNR (dB)")

    parser.add_argument("--pac_code", type=bool, default=False, help="target code: False => Polar code")
    parser.add_argument("--conv_poly", type=list[int], default=[1,0,1,1,0,1,1], help="convolutional polynomial")

    # Hyperparameters
    parser.add_argument("--graph_type", type=str, default='polar', help="graph type")
    parser.add_argument("--check_feature", type=str, default='default', help="check node feature")
    parser.add_argument("--v2c_aggr_type", type=str, default='mean', help="v2c aggregator type (sum, mean)")
    parser.add_argument("--conv_type", type=str, default='SAGEConv', help="check node to check node direction (SAGEConv, GATv2Conv)")
    # switching to info bit assignment
    parser.add_argument("--init_node_type", type=int, default=0, help="check node type initialized to: 0 frozen, 1 info")
    parser.add_argument("--c2c_direction", type=str, default='reversed', help="check node to check node direction (default, reversed)")
    # prefreeze and edge pruning
    parser.add_argument("--prefreeze_check_nodes", type=bool, default=False, help="prefreeze check nodes with low weights")
    parser.add_argument("--remove_edges", type=bool, default=False, help="remove_edges to prefrezed check nodes")
    parser.add_argument("--exclude_weight", type=int, default=-1, help="exclude check nodes with value or less")
    # range expansion
    parser.add_argument("--range_expansion", type=bool, default=True, help="reduced range initially then expand to full")
    parser.add_argument("--range_expansion_steps", type=int, default=4, help="takes n steps to full range")
    parser.add_argument("--range_expansion_end", type=float, default=0.25, help="end expansion at fraction of total training")
    # GAT experiment
    parser.add_argument("--add_self_loops", type=bool, default=False, help="add self loops to hetero graph")
    # reward shaping
    parser.add_argument("--anchor_first_info", type=bool, default=False, help="anchor first info bit to N-1")
    # conservative action selection
    parser.add_argument("--convservative_actions", type=bool, default=False, help="select convervatively (not always max)")
    parser.add_argument("--rho_action_sel", type=float, default=0.2, help="select from the (e.g 0.2) highest q(s,a) candidates")
    # New graph topology and arch
    parser.add_argument("--graph_arch", type=str, default='upo', help="Graph topology and GNN architecture (pccmp, gccmp, upo)")
    parser.add_argument("--gnn_arch", type=str, default='v2', help="GNN internal architecture (v1, v2)" )

    ################################################################################
    # Parameters for universal sequence
    ################################################################################
    parser.add_argument("--N_max", type=int, default=2048, help="largest code length (used in GNN and graph node feature gen)")
    parser.add_argument("--N_set", type=list[int], default=[64], help="target code lengths")
    parser.add_argument("--K_range", type=list[int], default=None, help="target code dimensions") # not used
    parser.add_argument("--K_min", type=int, default=1, help="target min dimension")
    parser.add_argument("--K_min_embed", type=int, default=None, help="Order restriction after K_min for lower-N embedding")
    parser.add_argument("--K_max", type=int, default=None, help="override target max dimension")
    parser.add_argument("--K_set_valid", type=list[int], default=None, help="validate code dimensions in set") # see below
#    parser.add_argument("--train_snr_offset", type=float, default=6.0, help="target snr offset from capacity limit (dB)")
    parser.add_argument("--approx_capacity_gap", type=bool, default=False, help="use approximate capacity gap")
    parser.add_argument("--auto_adjust", type=bool, default=True, help="auto adjust K with CRC length")
    parser.add_argument("--monitor_lower_n", type=bool, default=False, help="check for lower N order violations")
    parser.add_argument("--embed_lower_n", type=bool, default=False, help="Embedd lower N sequence")
    parser.add_argument("--promote_ln_node", type=bool, default=False, help="Promote lower N node")

    ################################################################################
    # Iterative learning
    ################################################################################
    parser.add_argument("--iterative_learning", type=bool, default=False, help="iterative learning")
    parser.add_argument("--resume_iterative_learning", type=bool, default=False, help="resume iterative learning")
    parser.add_argument("--lookahead_window", type=int, default=16, help="lookahead window")
    parser.add_argument("--min_initial_training", type=int, default=1000, help="minimum num episodes to train initial state")
    parser.add_argument("--settle_threshold", type=int, default=3, help="how many evals to wait before moving to next state")
    parser.add_argument("--freeze_set", type=int, default=1, help="number of K's to consider at each step")

    parser.add_argument("--disable_resume", action=ActionYesNo, default=False, help="do not load checkpoint")

    ################################################################################
    # Nested sequence
    ################################################################################
    parser.add_argument("--multi_n_training", type=bool, default=False, help="enable multi-N training")

    ################################################################################
    # Multi-stage training
    ################################################################################
    parser.add_argument("--multi_stage_training", type=bool, default=False, help="enable multi-stage training")
    parser.add_argument("--learned_seq", type=list[int], default=None, help="learned sequence")
    parser.add_argument("--K_start", type=int, default=1, help="pre-assign first K_start bits from learned sequence")
    parser.add_argument("--K_end", type=int, default=1, help="target max dimension")

    ################################################################################
    # UPO related
    ################################################################################
    parser.add_argument('--reverse', type=bool, default=True, help='assignment direction of synthetic bit channels: reverse := from N-1 to 0')
    parser.add_argument("--num_past_actions", type=int, default=8, help="retain this many past actions in the graph")
    parser.add_argument("--num_hops", type=int, default=1, help="increase action space to include nodes up to this number of hops")
    parser.add_argument('--feat_aggr', type=str, default="sum", help='feature aggregation (sum, concat)')

    ################################################################################
    # RL Training related parameters
    ################################################################################
    parser.add_argument('--fine_tune', type=bool, default=False, help='enable fine tuning')
    parser.add_argument('--query', type=bool, default=False, help='inference mode')
    parser.add_argument('--load_weights', type=bool, default=False, help="load weights")
    parser.add_argument('--load_model', type=bool, default=False, help="load model")

    parser.add_argument("--num_episodes", type=int, default=10000, help="num episodes to train network")
    parser.add_argument("--num_episodes_fine_tune", type=int, default=150, help="num episodes to train network for fine tuning")
    parser.add_argument("--train_snr_range", type=list[float], default=[0.0, 3.5], help="training snr range (dB)")

#    parser.add_argument("--batch_size", type=int, default=32, help="num transitions sampled from the replay buffer")
#    parser.add_argument("--gamma", type=float, default=None, help="discount factor") # see below
    parser.add_argument("--eps_start", type=float, default=1.0, help="start value of exploration rate")
    parser.add_argument("--eps_end", type=float, default=0.01, help="end value of exploration rate") # see below
    parser.add_argument("--eps_final_episode", type=int, default=10000, help="num episode to linearly anneal exploration rate to eps_end")
    parser.add_argument("--eps_fine_tune", type=float, default=0.01, help="value of exploration rate for fine tuning")
    parser.add_argument("--eps_decay_rate", type=float, default=0.999, help="exploration rate decay")
    parser.add_argument("--eps_decay_step", type=int, default=1, help="num steps before decay is applied")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="replay buffer size")
    parser.add_argument("--replay_start_size", type=int, default=1000, help="replay start size")
    parser.add_argument("--kappa", type=int, default=2, help="updates target policy every kappa episodes")
    parser.add_argument("--tau", type=float, default=0.0025, help="target policy soft update rate")

    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument("--max_err", type=int, default=500, help="BLER sim terminates after max errors")
    parser.add_argument("--max_runs", type=int, default=100000, help="BLER sim terminates after max runs")
    parser.add_argument("--max_err_valid", type=int, default=500, help="BLER sim terminates after max errors during validation")
    parser.add_argument("--num_runs", type=int, default=1000, help="num runs per CPU thread")
    parser.add_argument("--num_threads", type=int, default=8, help="num CPU cores/threads to use")

#    parser.add_argument("--lr", type=float, default=0.00025, help="optimizer learning rate")
    parser.add_argument("--lr_fine_tune", type=float, default=0.0001, help="optimizer learning rate for fine tuning")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="learning rate decay")
    parser.add_argument("--eps_adam", type=float, default=1.5e-4, help="eps added to the denominator for Adam")
    parser.add_argument("--amsgrad", type=bool, default=False, help="enable amsgrad variant")

    ################################################################################
    # DQL Enhancements
    ################################################################################
    # multistep learning
    parser.add_argument("--n_step", type=int, default=1, help="num steps in multistep learning")
    # prioritized experience replay
    parser.add_argument('--use_priority', type=bool, default=False, help="Enable prioritized experience replay")
    parser.add_argument("--alpha_per", type=float, default=0.5, help="how much prioritization is used, alpha = 0 uniform case")
    parser.add_argument("--beta_start_per", type=float, default=0.4, help="the amount of importance-sampling correction, beta = 1 fully compensate for non-uniform probabilities, anneal linearly to 1")
    parser.add_argument("--beta_end_per", type=float, default=1.0, help="the amount of importance-sampling correction, anneal linearly to value")
    parser.add_argument("--eps_per", type=float, default=1e-4, help="minimal priority, prevents zero probabilities")
    # duelling network
    parser.add_argument('--duelling_net', type=bool, default=False, help="enable duelling network configuration")
    parser.add_argument('--attend_to_frozen', type=bool, default=True, help="only consider frozen bits (current action space)")

    ################################################################################
    # PPO from SB3 framework
    ################################################################################
    parser.add_argument("--use_sb3", type=bool, default=True, help="Use stable-baselines3 library")
    parser.add_argument("--policy_algorithm", type=str, default="ppo", help="(ppo, spo)")

    parser.add_argument("--num_timesteps", type=int, default=100_000, help="num of training timesteps")
    parser.add_argument("--n_envs", type=int, default=8, help="num of parallel environments")
    parser.add_argument("--n_steps", type=int, default=64, help="horizon (T) in PPO")
    parser.add_argument("--n_epochs", type=int, default=20, help="num epochs to train using current rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="num transitions sampled from the rollout buffer")
    parser.add_argument("--lr", type=float, default=0.00025, help="optimizer learning rate")
    parser.add_argument("--clip_range", type=float, default=0.1, help="clip range for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor") # see below
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="weighting factor in generalized advantage estimation")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="entropy loss coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="gradient norm clipping limit")

    parser.add_argument("--show_progress", type=bool, default=False, help="")
    parser.add_argument("--save_freq", type=int, default=10_000, help="save model every this many timesteps")
    parser.add_argument("--eval_freq", type=int, default=1024, help="evaluate policy every this many timesteps")
    parser.add_argument("--n_eval_episodes", type=int, default=3, help="num of realizations from stochastic policy")
    parser.add_argument("--deterministic_eval", type=bool, default=False, help="")

    ################################################################################
    # paths
    ################################################################################
    parser.add_argument('--weights_path', type=str, default="models/pccmp_gnn_state.pt", help='weights')
    parser.add_argument('--model_path', type=str, default="models/a123456789_model.zip", help='model')
    parser.add_argument('--sequence_path', type=str, default="models/a123456789_seq.mat", help='sequence')
    parser.add_argument('--rate_prof_path', type=str, default="pccmp_gnn_rate_profile.mat", help='rate_profile')
    parser.add_argument('--rate_profs_path', type=str, default="pccmp_gnn_rate_profiles.mat", help='rate_profiles')
    parser.add_argument('--learned_seq_n128_path', type=str, default=None, help='sequence')
    parser.add_argument('--learned_seq_n256_path', type=str, default=None, help='sequence')
    parser.add_argument('--learned_seq_n512_path', type=str, default=None, help='sequence')
    parser.add_argument('--learned_seq_n1024_path', type=str, default=None, help='sequence')

    ################################################################################
    # output parameters
    ################################################################################
    parser.add_argument("--valid_interval", type=int, default=5, help="validate interval")
    parser.add_argument("--report_interval", type=int, default=100, help="report interval")
    parser.add_argument("--num_candidates", type=int, default=8, help="num candidates to save")
    parser.add_argument("--save_candidates_start", type=float, default=0.25, help="start saving candidates at fraction of training length e.g. 0.25*num_episodes")

    ################################################################################
    # PCCMP/GCCMP/UPO GNN parameters
    ################################################################################
    parser.add_argument("--d_x", type=int, default=1)
    parser.add_argument("--d_loc", type=int, default=4)
    parser.add_argument("--d_type", type=int, default=28)
    parser.add_argument("--d_init", type=int, default=64) # see below
    parser.add_argument("--n_node_type", type=int, default=4, help="node types = {V,I,F,P}, P = prefreezed")
    parser.add_argument("--d_hidden", type=int, default=64, help="hidden feature size")
    parser.add_argument("--n_conv_layers", type=int, default=3, help="num convolutional layers")
    parser.add_argument("--d_pool", type=int, default=1)
    parser.add_argument("--dv_hidden_mlp", type=list[int], default=[128, 32], help="MLP hidden layer dims")
    parser.add_argument("--d_z", type=int, default=1, help="graph node output dim")
    # GCCMP specific
    parser.add_argument("--d_type_gccmp", type=int, default=16)
    parser.add_argument("--n_node_type_gccmp", type=int, default=2, help="node types = {I,F}")
    parser.add_argument("--d_index", type=int, default=16)
    parser.add_argument("--n_node_index", type=int, default=None, help="number of nodes") # see below
    # Attention networks
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--att_dropout", type=float, default=0.0, help="attention dropout")
    parser.add_argument("--ffn_dropout", type=float, default=0.0, help="FFN dropout")
    parser.add_argument("--att_residual", type=bool, default=False, help="learnable residual connection in attentional networks (e.g. GATv2, MSAConv, Transformer)")
    parser.add_argument("--ffn_expand", type=int, default=2, help="expansion factor for FFN hidden layer")

    ################################################################################
    # Parallelization
    ################################################################################
    parser.add_argument('--data_parallel', type=bool, default=False, help="enable data parallel")
    parser.add_argument('--use_ddp', type=bool, default=True, help="enable distributed data parallel")
    parser.add_argument('--use_subproc_vec_env', type=bool, default=False, help="parallelize env with multiprocessing")
    parser.add_argument('--use_async_vec_env', type=int, default=0, help="parallelize env with async reward gen: 1 = multiprocessing, 2 = parabolt")

    ################################################################################
    # Misc parameters
    ################################################################################
#    parser.add_argument("--no_cuda", action=ActionYesNo, default=False, help="disables CUDA training")
#    parser.add_argument("--no_mps", action=ActionYesNo, default=False, help="disables mac GPU training")
    parser.add_argument("--use_cpu", action=ActionYesNo, default=False, help="force CPU only")
    parser.add_argument("--debug", action=ActionYesNo, default=False, help="debug mode")
    parser.add_argument('--compile', type=bool, default=False, help="compile model (torch 2.0)")
    parser.add_argument('--identity', type=str, default='007', help="task dependent id")

    parser.add_argument("--config", action="config")

    ################################################################################
    # Parse arguments
    ################################################################################

    cfg = parser.parse_args()

    ################################################################################
    # Derived parameters
    ################################################################################
    cfg.save_freq = cfg.num_timesteps // 10

    assert cfg.N_max >= np.max(cfg.N_set)

    if 'Transformer' in cfg.conv_type:
        cfg.d_loc *= 2
        cfg.d_type *= 2

    if cfg.d_init == None:
        cfg.d_init = cfg.d_loc + cfg.d_type

    cfg.n_node_index = cfg.N_set[-1] # max N

    if cfg.eps_end == None:
        cfg.eps_end = 1/(5*cfg.N)

    if cfg.gamma == None:
        cfg.gamma = 0.99**(1/cfg.n_step)
        print("n_step: ", cfg.n_step)
        print("gamma: ", cfg.gamma)

    lengths = [128, 256, 512, 1024]
    paths = ['learned_seq_n128_path', 'learned_seq_n256_path', 'learned_seq_n512_path', 'learned_seq_n1024_path']
    cfg.learned_seq_table = {}
    for length, path_attr in zip(lengths, paths):
        path = getattr(cfg, path_attr)
        if path is not None:
            print(f'loading {path_attr} at {path}')
            cfg.learned_seq_table[length] = sio.loadmat(path)

    # N dependent snr_offset
    cfg.train_snr_offsets = {}
    cfg.train_snr_offsets[64] = 6.0
    cfg.train_snr_offsets[128] = 6.0
    cfg.train_snr_offsets[256] = 6.0
    cfg.train_snr_offsets[512] = 5.75
    cfg.train_snr_offsets[1024] = 5.5
    cfg.train_snr_offsets[2048] = 5.25

    # K_max table
    cfg.K_max_dict = {}
    cfg.K_max_dict[64] = 40
    cfg.K_max_dict[128] = 80
    cfg.K_max_dict[256] = 200
    cfg.K_max_dict[512] = 300
    cfg.K_max_dict[1024] = 600
    cfg.K_max_dict[2048] = 1200

    if cfg.multi_n_training:
        print("Multi-N fine tuning")
        # fine-tuning mode
        # CR max ~ 0.8
        cfg.K_max_dict = {}
        cfg.K_max_dict[64] = 50
        cfg.K_max_dict[128] = 100
        cfg.K_max_dict[256] = 200
        cfg.K_max_dict[512] = 400
        cfg.K_max_dict[1024] = 800
        cfg.K_max_dict[2048] = 1600
        # PDCCH mode
        cfg.K_max_dict = {}
        cfg.K_max_dict[128] = 108
        cfg.K_max_dict[256] = 164
        cfg.K_max_dict[512] = 164
        cfg.K_max_dict[1024] = 164
        cfg.K_max_dict[2048] = 164
        # initialize with pre-trained model
        cfg.load_model = True
        print("Using model from:", cfg.model_path)

    if cfg.K_max is not None:
        assert len(cfg.N_set) == 1
        print(f'Overriding K_max at N {cfg.N_set[0]} to {cfg.K_max}')
        N = cfg.N_set[0]
        cfg.K_max_dict[N] = cfg.K_max

    print('K_max_dict', cfg.K_max_dict)

    def get_valid_set(N):
        if N == 64:
            K_set_valid = [20,32,40]
        elif N == 128:
            K_set_valid = [20,35,50,64,80]
#        elif N == 256:
#            K_set_valid = [20,40,60,80,100,128,140,160]
        else: # N > 128
            K_max = cfg.K_max_dict[N]
            K_set_valid = np.linspace(20,K_max,8).astype(int).tolist()
        return K_set_valid

    cfg.K_valid_dict = {}
    for N in cfg.N_set:
        cfg.K_valid_dict[N] = get_valid_set(N)

    print('K_valid_dict', cfg.K_valid_dict)

    if cfg.pac_code:
        cfg.crc_poly = [] # disable crc for now
        print("PAC code training");
        print("crc poly: ", cfg.crc_poly)
        print("conv poly: ", cfg.conv_poly)
        print("graph type: ", cfg.graph_type)
        print("list size: ", cfg.L)
    else:
        print("Polar code training");
        max_N = cfg.N_set[-1]
        crc_size = len(cfg.crc_poly) - 1 if len(cfg.crc_poly) > 0 else 0

        # auto adjust K values
        if cfg.genie_aided:
            cfg.K_min = max(cfg.K_min, int(np.log2(cfg.L))+1)
        else:
            cfg.K_min = max(cfg.K_min, crc_size+1)
        if cfg.auto_adjust:
            for N in cfg.K_max_dict:
                cfg.K_max_dict[N] = min(cfg.K_max_dict[N] + crc_size, N)
            for N in cfg.N_set:
                cfg.K_valid_dict[N] = [min(K + crc_size, N) for K in cfg.K_valid_dict[N]]

        print("crc_size", crc_size)
        print('effective K_min', cfg.K_min)
        print('effective K_max_dict', cfg.K_max_dict)
        print('effective K_valid_dict', cfg.K_valid_dict)

        if cfg.genie_aided:
            print("Genie-aided SCL enabled");
            cfg.crc_poly = [] # disable crc

    if cfg.iterative_learning:
        print("Iterative learning")

    if cfg.multi_stage_training:
        print("Multi stage training")
        assert cfg.load_model == True
        assert len(cfg.N_set) == 1
        print("Using model from:", cfg.model_path)
        print(f"K_start, K_end: {cfg.K_start}, {cfg.K_end}")
        K_set_valid = np.linspace(cfg.K_start+1,cfg.K_end,8).astype(int).tolist()
        N = cfg.N_set[-1]
        cfg.K_valid_dict[N] = K_set_valid
        print('K_valid_dict', cfg.K_valid_dict)

    if cfg.graph_arch == 'gccmp':
        print("Using GCCMP topo/arch")
        cfg.d_type = cfg.d_type_gccmp
        cfg.n_node_type = cfg.n_node_type_gccmp
        cfg.n_node_index = cfg.N_set[-1] # max N

    if cfg.query:
        print('Query mode')
        cfg.load_model = True
        cfg.num_timesteps = 0
#        cfg.load_weights = True
#        cfg.num_episodes = 0

#    if cfg.fine_tune:
#        print('Fine tuning')
#        cfg.load_weights = True
#        cfg.train_snr_range = [cfg.target_snr] * 2
#        cfg.eps_start = cfg.eps_fine_tune
#        cfg.eps_end = cfg.eps_fine_tune
#        cfg.lr = cfg.lr_fine_tune
#        cfg.num_episodes = cfg.num_episodes_fine_tune
#        cfg.valid_interval = 5
#        cfg.report_interval = 5

    if cfg.debug:
        print('Debug mode')
        cfg.disable_resume = False
        cfg.batch_size = 32
        cfg.n_steps = 32
        cfg.num_candidates = 3
        cfg.eval_freq = 512
        cfg.num_timesteps = cfg.eval_freq * 8
        cfg.save_freq = 1024
        cfg.num_episodes = 10 * cfg.n_envs
        cfg.range_expansion_end = 0.8
#        cfg.replay_start_size = 32
#        cfg.num_episodes = 10
#        cfg.valid_interval = 1
#        cfg.report_interval = 1
#        cfg.eps_final_episode = 5

    print(cfg)

    return cfg
