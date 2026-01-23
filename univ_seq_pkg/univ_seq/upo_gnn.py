#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np

import torch
import torch.nn as nn
from torch.linalg import vector_norm

from torch_geometric.nn import SAGEConv, GATv2Conv, ResGatedGraphConv, LayerNorm
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_mean_pool
from torch.nn.functional import gelu

from torch.nn.parallel import DistributedDataParallel as DDP
from stable_baselines3.common.policies import BasePolicy

################################################################################
# define Homogeneous-GNN for UPO based learning
################################################################################
class Init(nn.Module):
    r"""INIT operator with binary representation embedding
    """
    def __init__(self, d_init, n_node_type, d_bits, aggr='sum'):
        # d_bits := log2(N_max)
        super().__init__()
        print('feature aggr', aggr)

        d_out = d_init if aggr == 'sum' else d_init//2 # preserve output dimension
        self.q_embedding = nn.Embedding(n_node_type, d_out) # type embedding
        self.b_embedding = nn.Linear(d_bits, d_out, bias=False) # binary rep embedding
        if aggr == 'sum':
            self.aggregate = lambda x,y: x+y
        elif aggr == 'concat':
            self.aggregate = lambda x,y: torch.cat((x,y), dim=-1)
        else:
            raise Exception('sum or concat only')

    def forward(self, x, node_type):

        bin_rep = x
        b = self.b_embedding(bin_rep)
        q = self.q_embedding(node_type)
        h = self.aggregate(b,q)

        return h


class Aggregator(nn.Module):
    r"""AGG operator
    """
    def __init__(self, en_act=True):
        super().__init__()
        self.en_act = en_act

    def forward(self, x):

        # 2-norm normalization
        norm = vector_norm(x, dim=-1, keepdim=True)
        norm_inv = norm.pow(-1)
        x = x * norm_inv
        # apply relu
        if self.en_act:
            x = x.relu()

        return x


class MLP(nn.Module):
    def __init__(self, d_in, dv_hidden, d_out):
        super().__init__()

        assert len(dv_hidden) == 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in,dv_hidden[0]))
        self.layers.append(nn.Linear(dv_hidden[0],dv_hidden[1]))
        self.linear = nn.Linear(dv_hidden[1],d_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        x = self.linear(x)
        return x


################################################################################
# Actor critic model
################################################################################
class ActorCriticReadOut(nn.Module):
    def __init__(self, d_hidden, d_pool, dv_hidden_mlp, d_z, multi_length=False):
        super().__init__()

        print('multi_length', multi_length)

        d_in_mlp = d_hidden
        self.mlp = MLP(d_in_mlp, dv_hidden_mlp, d_z)

        self.vf_head = MLP(d_hidden, dv_hidden_mlp, 1)

        self.d_in_mlp = d_in_mlp
        self.d_hidden = d_hidden
        self.d_z = d_z
        self.multi_length = multi_length

    def forward(self, x, node_type, action_mask, node_indices, batch, N):
        # Note: remove global pooling for individual head

        # global pooling operation
        h_mean = global_mean_pool(x, batch) # [n_batch, n_features]

        # final MLP processing
        # use gather-scatter to reduce compute
        indices_s = torch.nonzero(action_mask).squeeze()
        indices_g = indices_s.view(-1,1).expand(-1,x.shape[-1])

        x_subset = x.gather(0, indices_g)
        z_subset = self.mlp(x_subset).squeeze()

        z_all = x.new_full((x.shape[0],), -torch.inf)
        z_all.scatter_(0, indices_s, z_subset)

        # handle unequal graph sizes
        # N = [N0, N1, ..., N7]
        if batch is not None:
            if self.multi_length:
                N_max = torch.max(N)
                batch_size = N.shape[0]
                indices = node_indices + (N_max * batch)
                z_len = N_max * batch_size
                z = z_all.new_full((z_len,), -torch.inf)
                z.scatter_(0, indices, z_all)
                z = z.view(-1, N_max)
            else:
                z = z_all.view(-1, N[0])
        else:
            z = z_all
        # z := logits for pi(a|s)


        # V(s) head
        v = self.vf_head(h_mean)

        return z, v


class FFN(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout = 0.0, expand = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expand * n_embd),
            nn.ReLU(),
            nn.Linear(expand * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


################################################################################
# GNN arch V2: pre-LN and residual connection
# each layer consists of the following:
#        LN -> NA -> activation
################################################################################
class UPOGNNv2ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        N_max = cfg.N_max
        d_bits = int(np.log2(N_max))
        self.init = Init(cfg.d_init, cfg.n_node_type, d_bits, cfg.feat_aggr)
        self.convs = nn.ModuleList()
        self.lnorms = nn.ModuleList()
        self.lnorms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.add_ffn = True if cfg.conv_type == 'Transformer' else False
        print(f'conv_type {cfg.conv_type}')
        for iLayer in range(cfg.n_conv_layers):
            d_in = cfg.d_init if iLayer == 0 else cfg.d_hidden
            self.lnorms.append(LayerNorm(d_in))
            self.lnorms2.append(LayerNorm(d_in) if cfg.conv_type == 'Transformer' else None)
            self.ffns.append(FFN(cfg.d_hidden, dropout=cfg.ffn_dropout, expand=cfg.ffn_expand) if cfg.conv_type == 'Transformer' else None)
            if cfg.conv_type == 'SAGEConv':
                self.convs.append(SAGEConv(d_in, cfg.d_hidden, aggr='mean'))
            elif cfg.conv_type == 'GATv2Conv':
                self.convs.append(GATv2Conv(d_in, cfg.d_hidden//cfg.n_heads, heads=cfg.n_heads, add_self_loops=False, residual=cfg.att_residual))
            elif cfg.conv_type == 'ResGatedGraphConv':
                self.convs.append(ResGatedGraphConv(d_in, cfg.d_hidden))
            else:
                raise Exception('Unknown convolutional operator')
        self.post_proc = ActorCriticReadOut(cfg.d_hidden, cfg.d_pool, cfg.dv_hidden_mlp, cfg.d_z, len(cfg.N_set) > 1)

    def forward(self, data):

        # INIT
        h = self.init(data.x, data.node_type)
        # GNN processing
        for ln, conv, ln2, ffn in zip(self.lnorms, self.convs, self.lnorms2, self.ffns):
            h_in = h
            h = ln(h)
            h = conv(h, data.edge_index)
            h = h.relu() if not self.add_ffn else h
            h = h + h_in
            # FFN in transformer
            if self.add_ffn:
                h_in = h
                h = ln2(h)
#                h = h.relu() # add RELU
                h = ffn(h)
                h = h + h_in

        # Post processing
        batch = data.batch if isinstance(data, Batch) else None
        z, v = self.post_proc(h, data.node_type, data.action_mask, data.node_indices, batch, data.N)
        # Return logits for pi(a|s) and V(s)
        return z, v


################################################################################
# Initial GNN arch: conv layer with normalization
################################################################################
class UPOGNNActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        N_max = cfg.N_max
        d_bits = int(np.log2(N_max))
        self.init = Init(cfg.d_init, cfg.n_node_type, d_bits, cfg.feat_aggr)
        self.convs = nn.ModuleList()
        print(f'conv_type {cfg.conv_type}')
        for iLayer in range(cfg.n_conv_layers):
            d_in = cfg.d_init if iLayer == 0 else cfg.d_hidden
            if cfg.conv_type == 'SAGEConv':
                self.convs.append(SAGEConv(d_in, cfg.d_hidden, aggr='mean'))
            else:
                raise Exception('Unknown convolutional operator')
        self.aggr = Aggregator()
        self.post_proc = ActorCriticReadOut(cfg.d_hidden, cfg.d_pool, cfg.dv_hidden_mlp, cfg.d_z, len(cfg.N_set) > 1)

    def forward(self, data):

        # INIT
        h = self.init(data.x, data.node_type)
        # GNN processing
        for conv in self.convs:
            h = conv(h, data.edge_index)
            h = self.aggr(h)
        # Post processing
        batch = data.batch if isinstance(data, Batch) else None
        z, v = self.post_proc(h, data.node_type, data.action_mask, data.node_indices, batch, data.N)
        # Return logits for pi(a|s) and V(s)
        return z, v

# create a policy model that conforms to stable-baselines3 API
import torch.optim as optim
from torch.distributions import Categorical
class GnnActorCriticPolicy(BasePolicy):
    """
    Provide high level API for now
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        use_sde = False,
        cfg = None,
        device = None,
    ):
        # use cfg for all configuration needs for now
        assert cfg is not None
        assert device is not None

        super().__init__(
            observation_space,
            action_space)

        # set up model instance
        print('gnn_arch', cfg.gnn_arch)
        if cfg.gnn_arch == 'v1':
            self.model = UPOGNNActorCritic(cfg)
        elif cfg.gnn_arch == 'v2':
            self.model = UPOGNNv2ActorCritic(cfg)
        else:
            raise Exception('Unknown GNN architecture')
        self.model.to(device)
        if cfg.use_ddp:
            self.model = DDP(self.model, device_ids=[device])
        print(self.model)

        # FIXME: find the right parameters for Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_schedule(1))

    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass
        """
        latent_pi, values = self.model.forward(obs)
        distribution = Categorical(logits=latent_pi)
        if deterministic:
            actions = distribution.mode
        else:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        latent_pi, values = self.model.forward(obs)
        distribution = Categorical(logits=latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def extract_features(self, obs, features_extractor = None):
        raise Exception("Not implemented")

    def get_distribution(self, obs):
        latent_pi, values = self.model.forward(obs)
        distribution = Categorical(logits=latent_pi)
        return distribution

    def _predict(self, observation, deterministic: bool = False):
        distribution = self.get_distribution(observation)
        if deterministic:
            actions = distribution.mode
        else:
            actions = distribution.sample()
        return actions

    def predict(self, obs, state, episode_start, deterministic: bool = False):
        latent_pi, values = self.model.forward(obs)
        distribution = Categorical(logits=latent_pi)
        if deterministic:
            actions = distribution.mode
        else:
            actions = distribution.sample()
        return actions, None

    def predict_values(self, obs):
        latent_pi, values = self.model.forward(obs)
        return values

    def reset_noise(self, n_envs = 1):
        # do nothing
        pass

    def set_training_mode(self, mode: bool):
        self.train(mode)

if __name__ == "__main__":

    from pccmp_graph import gen_pccmp_graph

    N = 4
    SNRdB = 0.0
    fbm = [0,1,0,1]

    data = gen_pccmp_graph(N, SNRdB, fbm)

    print(data)

    print(data.x_dict)
    print(data.node_type_dict)
    print(data.edge_index_dict)

    d_x = 1
    d_loc = 4
    d_type = 28
    n_node_type = 3
    d_hidden = 64
    num_layers = 3
    d_pool = 1
    dv_hidden_mlp = (128, 32)
    d_z = 1

    init_layer = Init(d_x, d_loc, d_type, n_node_type)
    out_dict = init_layer(data.x_dict, data.node_type_dict)
    print(out_dict)

    d_init = d_loc + d_type
    d_hidden = 64
    num_layers = 3

    x = torch.Tensor([[1,1,1,1],
                      [1,0,1,0]])
    x_dict = {}
    x_dict['var'] = x
    print(x_dict)
    agg = Aggregator(en_act=False)
    out_dict = agg(x_dict)
    print(out_dict)


    model = PCCMPGNN(d_x, d_loc, d_type, n_node_type,
                     d_hidden, num_layers,
                     d_pool, dv_hidden_mlp)

    print(model)

    with torch.no_grad():  # Initialize lazy modules.
        z = model(data.x_dict, data.node_type_dict, data.edge_index_dict, theta)

    z = model(data.x_dict, data.node_type_dict, data.edge_index_dict, theta)

    # dump parameters
    for name, param in model.convs[0].named_parameters():
        print (name, param.data)
    print(z)

    # test batching
    N = 4
    K = 2
    SNRdB = 0.0
    fbm = [0,1,0,1]

    data = gen_pccmp_graph(N, SNRdB, fbm)



