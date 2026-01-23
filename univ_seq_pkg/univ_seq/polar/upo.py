#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
from itertools import combinations
from .sequences import int2bin, get_beta_exp_seq, get_nr_seq
import argparse

# helper functions
class Space:
    def __init__(self, seq):
        self.seq = seq

    def __str__(self):
        return f"{len(self.seq)} {self.seq}"

def get_combinations(arr, r):
    return np.array(list(combinations(arr,r)))

def check_addition(bit_mat, bit_mat_cpl, n_bits, N):
    cond = np.full((N,N), True, dtype=bool)
    for ibit in range(n_bits):
        bit_vec = bit_mat[:,[ibit]] # preserve dim
        cond &= bit_vec < bit_vec.T
        cond = np.triu(cond)
        #print(cond)
    indices = np.argwhere(cond)
    def check_equal_cpl(i,j):
        return np.all(bit_mat_cpl[i,:] == bit_mat_cpl[j,:])
    indices = [pair for pair in indices if check_equal_cpl(pair[0], pair[1])]
    indices = np.array(indices)
    print('find relations')
    print(indices)
    return indices

def check_left_swap(bit_mat, bit_mat_cpl, n_bits, N):
    cond = np.full((N,N), True, dtype=bool)
    bit_vec = bit_mat[:,[0]] # preserve dim
    cond &= bit_vec < bit_vec.T
    cond = np.triu(cond)
    bit_vec = bit_mat[:,[1]] # preserve dim
    cond &= bit_vec > bit_vec.T
    cond = np.triu(cond)
    indices = np.argwhere(cond)
    def check_equal_cpl(i,j):
        return np.all(bit_mat_cpl[i,:] == bit_mat_cpl[j,:])
    indices = [pair for pair in indices if check_equal_cpl(pair[0], pair[1])]
    indices = np.array(indices)
    print('find relations')
    print(indices)
    return indices

# see article
# He, Gaoning, Jean-Claude Belfiore, Ingmar Land, Ganghua Yang, Xiaocheng Liu, Ying Chen, Rong Li et al. "Beta-expansion: A theoretical framework for fast and recursive construction of polar codes." In GLOBECOM 2017-2017 IEEE Global Communications Conference, pp. 1-6. IEEE, 2017.

class UPO:
    def __init__(self, N, reverse=False, lower_n_seq=None, promote_ln_node=False, K_min=None):
        self.N = N
        self.reverse = reverse

        # embed lower N
        print('Embed lower N sequence') if lower_n_seq is not None else None
        print('Promote lower N node') if promote_ln_node else None
        print(f'K_min {K_min}') if lower_n_seq is not None else None
        self.embed_lower_n = True if lower_n_seq is not None else False
        if self.embed_lower_n:
            self.lower_n_seq = lower_n_seq
            self.unused_ln_nodes = lower_n_seq
            self.promote_ln_node = promote_ln_node
            self.promoted_nodes = np.array([], dtype=int)
            self.init_step = True
            self.K_min = K_min
            self.lower_N = lower_n_seq.size
            assert self.lower_N == N//2

        self.relations = self.find_upo_relations(N)
        self.neighbor_list, self.widths = self.list_neighbors(self.relations, N)
        self.nbl_len = len(self.neighbor_list)

        # states for traversing UPO trellis/graph
        if not reverse:
            self.nbl_idx_start = 0
            self.nbl_idx = self.nbl_idx_start
            self.inc_dir = 1 # increment direction
            self.nbl_idx_end = self.nbl_len - 1
            self.bit_idx = 0
            self.seq_idx = 0
        else: # reverse
            self.nbl_idx_start = self.nbl_len - 1
            self.nbl_idx = self.nbl_idx_start
            self.inc_dir = -1 # increment direction
            self.nbl_idx_end = 0
            self.bit_idx = N - 1
            self.seq_idx = N - 1

    def find_upo_relations(self, N_max, debug=False):
        # find universal partial order recursively
        # relations (i,j) <=> i < j
        # UPO - addition and left-swap
        n_max = int(np.log2(N_max))
        relations = []
        for n in range(1, n_max+1):
            print('n', n) if debug else None
            N = 2**n
            if n == 1:
                relations = np.array([0,1]).reshape(1,2)
                continue
            # copy relations + N/2
            relations_o = relations
            relations_c = relations + 2**(n-1)
            relations = np.concatenate((relations, relations_c))
            print('relations+c') if debug else None
            print(relations) if debug else None
            # new order
            print('new order') if debug else None
            bin_mat = int2bin(np.arange(N), n)
    #        for bit_idx in range(1, n):
            for bit_idx in range(1, 2): # check adjacent bit index only, see comment below
                #check_new_order()
                cond = np.full((N,N), True, dtype=bool)
                bit_vec = bin_mat[:,[0]]
                cond &= bit_vec < bit_vec.T
                cond = np.triu(cond)
                bit_vec = bin_mat[:,[bit_idx]] # preserve dim
                cond &= bit_vec > bit_vec.T
                cond = np.triu(cond)
                indices = np.argwhere(cond)
                # check cpl
                n_vec = np.arange(n)
                bit_idx_pair = np.array([0, bit_idx])
                bit_idx_cpl = np.setdiff1d(n_vec, bit_idx_pair)
                bit_mat_cpl = bin_mat[:,bit_idx_cpl]
                def check_equal_cpl(i,j):
                    return np.all(bit_mat_cpl[i,:] == bit_mat_cpl[j,:])
                indices = [pair for pair in indices if check_equal_cpl(pair[0], pair[1])]
                indices = np.array(indices)

    #            # check chain
    #            def check_chain(i,j):
    #                # this only check for one-hop relations
    #                # i.e. x < z => x < y < z
    #                # however, we can have 2 or more hops but this is known
    #                # we can remove that by only counting the adjacent bix_idx
    #                # i.e. bit_idx = 1 only, not 1:n
    #                indices = relations[:,0] == i
    #                lhs = relations[indices,1]
    #                indices = relations[:,1] == j
    #                rhs = relations[indices,0]
    #                its = np.intersect1d(lhs, rhs)
    #                return True if its.size > 0 else False
    #            indices = [pair for pair in indices if not check_chain(pair[0], pair[1])]
    #            indices = np.array(indices)

                print('found relations') if debug else None
                print(indices) if debug else None
                relations = np.concatenate((relations_o, indices, relations_c))
            print(f'relations n{n}') if debug else None
            print(relations) if debug else None
        return relations

    def list_neighbors(self, relations, N):
        # list neigbhors as in fig.1 of paper
        # relations (x,y) => x < y
        index = 0
        bit_set = 0 # start
        neighbor_list = []
        neighbor_list.append([0])
        widths = []
        widths.append(1)
        while index < (N-1):
            indices = np.where(np.isin(relations[:,0], bit_set))[0]
            # get children
            children = relations[indices,1]
            children = np.unique(children)
            neighbor_list.append(children.tolist())
            widths.append(children.size)
            bit_set = children
            index = np.min(bit_set)

        return neighbor_list, widths

    def is_children(self, lhs, rhs):
        relations = self.relations
        reverse = self.reverse
        lhs_idx = 0 if not reverse else 1
        rhs_idx = 1 if not reverse else 0
        indices = np.where(np.isin(relations[:, lhs_idx], lhs))[0]
        children = relations[indices, rhs_idx]
        return np.isin(rhs, children)

    def get_children(self, nodes, constraint_set=None):
        # get children of nodes, contained inside the constraint_set
        relations = self.relations
        reverse = self.reverse
        lhs_idx = 0 if not reverse else 1 # parent
        rhs_idx = 1 if not reverse else 0 # child
        indices = np.where(np.isin(relations[:, lhs_idx], nodes))[0]
        children = relations[indices, rhs_idx]
        children = np.unique(children)
        if constraint_set is not None:
            indices = np.where(np.isin(children, constraint_set))[0]
            children = children[indices]
        return children

    def reset(self):
        self.nbl_idx = self.nbl_idx_start
        self.children = np.array(self.neighbor_list[self.nbl_idx])
        self.unused_nodes = self.children
        self.used_nodes = np.array([], dtype=int)
        if self.embed_lower_n:
            self.unused_ln_nodes = self.lower_n_seq
            self.init_step = True

    def fast_forward(self, seq, K_start, num_hops=1):
#        print('seq', seq)
        for k in range(K_start):
            actions = self.get_actions(num_hops=num_hops)
#            print('actions', actions)
            action = seq[-(k+1)]
#            print('action', action)
            self.update_action(action)

    def get_actions_and_scope(self, num_hops=1, num_past_actions=0, eval_mode=False):
        action_space = self.get_actions(num_hops, eval_mode)
        # construct scope
        if num_past_actions > 0:
            past_actions = self.used_nodes[-num_past_actions:]
            scope = np.concatenate((past_actions, action_space))
        else:
            scope = action_space
#        print('actions', Space(self.actions))
#        print('scope', scope)
        return self.actions, scope

    def get_actions(self, num_hops=1, eval_mode=False):
        num_hops_in = num_hops
        assert num_hops <= 2, "need to add support for 3 or more hops"
        unused_nodes = self.unused_nodes
        nodes = unused_nodes[ np.logical_not(self.is_children(unused_nodes, unused_nodes)) ]
        action_space = nodes
        num_hops -= 1
        while num_hops > 0:
            nodes = self.get_children(nodes, constraint_set=unused_nodes)
            action_space = np.concatenate((action_space, nodes))
            num_hops -= 1
        if self.embed_lower_n and action_space.size > 0:
            if self.init_step:
                # order is undetermined initially
                n_restricted = self.lower_N - self.K_min
                restricted_set = self.unused_ln_nodes[:n_restricted]
            else:
                # restrict actions to guarantee order in lower N
                next_ln_node = self.unused_ln_nodes[-1]
                restricted_set = self.unused_ln_nodes[:-1]
            indices = np.isin(action_space, restricted_set)
            bad_actions = action_space[indices]
            if bad_actions.size > 0:
#                print(f'removing actions {bad_actions} from action_space {action_space}: lower N violation') if not eval_mode else None
                action_space = np.setdiff1d(action_space, bad_actions)
                # promote action only if it is 2nd hop in the unused_nodes set
                if self.promote_ln_node and not self.init_step:
                    assert num_hops_in == 1
                    second_hop_nodes = self.get_children(nodes, constraint_set=unused_nodes)
                    if not next_ln_node in action_space and next_ln_node in second_hop_nodes:
                        self.promoted_nodes = np.union1d(self.promoted_nodes, next_ln_node)
                        action_space = np.append(action_space, next_ln_node)
                        print(f'promoted next_ln_node {next_ln_node} to action space') if not eval_mode else None
        self.actions = action_space
        return self.actions

    def update_action(self, action):
        assert action in self.actions, 'action outside of permissible action space'
        self.unused_nodes = np.setdiff1d(self.unused_nodes, [action])
        self.used_nodes = np.concatenate((self.used_nodes, [action]))
        if action in self.children and not self.nbl_idx == self.nbl_idx_end:
            self.nbl_idx += self.inc_dir
            self.children = np.array(self.neighbor_list[self.nbl_idx])
            self.unused_nodes = np.concatenate((self.unused_nodes, self.children))
        if self.embed_lower_n:
            if self.init_step and action in self.unused_ln_nodes:
                # guaranteed to be in the first K_min candidates
                index = self.unused_ln_nodes == action
                self.unused_ln_nodes = np.delete(self.unused_ln_nodes, index)
                # exit out of init_step
                n_restricted = self.lower_N - self.K_min
                if self.unused_ln_nodes.size <= n_restricted:
                    self.init_step = False
            else:
                if action == self.unused_ln_nodes[-1]:
                    self.unused_ln_nodes = self.unused_ln_nodes[:-1]
        return self.unused_nodes.size == 0

    def traverse_graph(self, take_ln_action=False):
        self.reset()
        terminated = False
        action_space_vec = []
        while not terminated:
            actions = self.get_actions()
            action_space_vec.append(actions.size)
            action = np.random.choice(actions)
            if self.embed_lower_n and take_ln_action:
                next_ln_node = self.unused_ln_nodes[-1]
                if next_ln_node in actions:
                    print(f'***taking lower N decision')
                    action = next_ln_node
            terminated = self.update_action(action)
            print('choices', Space(actions), 'decision', action)
        self.action_space_vec = action_space_vec

    def list_decisions(self, seq):
        '''
        1. Walk along the sequence, examine the decision space based on previous decisions
        2. Check number of rule violations from a sequence
        3. Walk in the reverved order, i.e. decision walking from N-1 to 0
        '''
        neighbor_list = self.neighbor_list
        relations = self.relations
        N = self.N
        reverse = self.reverse

        nbl_idx = self.nbl_idx
        inc_dir = self.inc_dir
        nbl_idx_end = self.nbl_idx_end
        bit_idx = self.bit_idx
        seq_idx = self.seq_idx
        nbl_len = self.nbl_len

        ntimes = 1
        assert seq.size == N

        # first action
        assert len(neighbor_list[nbl_idx]) == 1 and neighbor_list[nbl_idx][0] == bit_idx
        assert seq[seq_idx] == bit_idx
        used_nodes = np.array([bit_idx])
        nbl_idx += inc_dir
        children = np.array(neighbor_list[nbl_idx])
        unused_nodes = children
        print('start', seq[seq_idx])

        for seq_idx in range(1,N):
            bit = seq_idx if not reverse else (N-1) - seq_idx
            blanket = unused_nodes
            choices = blanket[ np.logical_not(self.is_children(blanket, blanket)) ]
            decision = seq[bit]
            # terminate early
            if decision == -1:
                break
    #        print('bit', bit)
            print('choices', Space(choices), 'decision', decision)

            if not decision in choices:
                print(f'violated rule {ntimes} time(s)!')
                ntimes += 1
                #print('unused', unused_nodes)
                # locate wrong decision
                if decision in unused_nodes:
                    print('- decision in unused set, ok')
                    # check in which neighborhood the decision made
                    level = 0
                    nodes_at_level = choices
                    decision_space = choices
                    while decision not in nodes_at_level:
                        level += 1
                        nodes_at_level = self.get_children(nodes_at_level, constraint_set=unused_nodes)
                        decision_space = np.concatenate((decision_space, nodes_at_level))
                    print(f'- decision in level({level}) neighborhood, {level+1} hops away')
                    print('- effective decision space', Space(decision_space))
    #                print('unused set', Space(unused_nodes))
                while not decision in unused_nodes and not nbl_idx == nbl_idx_end:
                    print('- decision not in unused set!!!')
                    # load next set of children
                    nbl_idx += inc_dir
                    children = np.array(neighbor_list[nbl_idx])
                    unused_nodes = np.concatenate((unused_nodes, children))

            unused_nodes = np.setdiff1d(unused_nodes, [decision])
            used_nodes = np.concatenate((used_nodes, [decision]))
            if decision in children and not nbl_idx == nbl_idx_end:
                nbl_idx += inc_dir
                children = np.array(neighbor_list[nbl_idx])
                unused_nodes = np.concatenate((unused_nodes, children))
    #        print('used_nodes', used_nodes)
    #        print('unused_nodes', unused_nodes)


    def check_upo_violations(self, seq):
        relations = self.relations
        print('Checking UPO validations')
        lhs = np.argwhere(seq_N == relations[:,[0]])[:,1]
        rhs = np.argwhere(seq_N == relations[:,[1]])[:,1]
        upo_cond = lhs < rhs
        idx = np.where(np.logical_not(upo_cond))
        print('relations violated')
        print(relations[idx])
        print('UPO?', np.all(lhs < rhs))


if __name__ == "__main__":

#    parser = argparse.ArgumentParser(description='BLER sweep')
#
#    # Add arguments
#    parser.add_argument('--num_threads', type=int, default=8, help='CPU cores')
#    parser.add_argument('--runs_per_thread', type=int, default=1000, help='num trials per thread')
#    parser.add_argument('--use_seq', type=str, help='simulate BLER for sequence (beta,nr)')
#
#    # Parse the arguments
#    args = parser.parse_args()

    test_beta = True
    test_nr = True
    test_prefreeze = False
    reverse = True
    N = 256

    print('reverse', reverse)
    print('N', N)

    # test embedding lower N sequence
    import scipy.io as sio
    seq_file = 'init_model/q2d5f54ddj_n128_learned_seq.mat'
    data = sio.loadmat(seq_file)
    learned_seq_N = data['learned_seq'].squeeze()
    print('learned_seq', learned_seq_N)
    upo = UPO(N, reverse, learned_seq_N, promote_ln_node=True, K_min=16)
    upo.traverse_graph(take_ln_action=False)
    print('promoted_nodes', upo.promoted_nodes)

    exit()

    upo = UPO(N, reverse)
    relations = upo.relations
#    print('relations')
#    print(relations)
    neighbor_list, widths = upo.neighbor_list, upo.widths
    print('neighbor_list')
    print(neighbor_list)
    print('width')
    print(widths)

    if test_prefreeze:
        # test shortening pre-freezing
        AL = 4
        E = AL*108
        U = N - E

        prefreeze_indices = np.arange(N)
        prefreeze_indices[:E] = -1
        print('prefreeze_indices', prefreeze_indices)
        upo.list_decisions(prefreeze_indices)

    if test_beta:
        seq = get_beta_exp_seq()
        seq_N = seq[seq < N]
        print('beta')
        print(seq_N)
        upo.list_decisions(seq_N)

    if test_nr and N <= 1024:
        seq = get_nr_seq()
        seq_N = seq[seq < N]
        print('NR')
        print(seq_N)
        upo.list_decisions(seq_N)
        upo.check_upo_violations(seq_N)

    # random traversal
    print('random traversal')
    upo.traverse_graph()

    # compute size of state space
    #print(upo.action_space_vec)
    prod : int = 1
    for elem in upo.action_space_vec:
        prod *= elem
    print(f'prod {prod}')
    print(f'num_digits {len(str(prod))}')

    exit()

#    seq = get_nr_seq()
    seq = get_beta_exp_seq()

    N_vec = np.array([64,128,256])
    N_vec = np.array([64,128,256,512,1024])
    N_vec = np.array([64])

    # check nesting properties
    print('nesting properties')
    for N in N_vec:
        print(f'N={N}')
        seq_N = seq[seq < N]

        N0 = N//2
        seq_lo = seq_N[seq_N < N0]
        seq_hi = seq_N[seq_N >= N0]
        seq_hi_ = seq_hi - N0

        print('seq_lo :', seq_lo)
        print('seq_hi_:', seq_hi_)
        print('equal?', np.array_equal(seq_lo, seq_hi_))

