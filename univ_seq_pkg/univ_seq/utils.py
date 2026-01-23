#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import sys
import numpy as np
import torch
import random

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class PriorityQueue:
    def __init__(self, size):
        self.size = size
        self.list = []

    def add(self, element):
        bler_vec, fbm = element
        this_list = self.list

        # reject if list full
        if len(this_list) == self.size:
            last_bler_vec, _ = this_list[-1]
            if np.mean(bler_vec) > np.mean(last_bler_vec):
                print('Candidate rejected')
                return
        # check unique
        fbms = [ x[1] for x in this_list ]
        found = np.any( [ np.array_equal(fbm, x) for x in fbms ] )
        if not found:
            this_list.append(element)
            sorted_list = sorted(this_list, key=lambda x: np.mean(x[0]), reverse=False)
            # trim len
            if len(sorted_list) > self.size:
                sorted_list.pop()
            # locate fbm
            fbms = [ x[1] for x in sorted_list ]
            index = np.argwhere( [ np.array_equal(fbm, x) for x in fbms ] ).squeeze()
            print(f'Inserted new candidate at pos {index} of {len(sorted_list)}')
            self.list = sorted_list
        else:
            print('Candidate already in list')

        # return true if new candidate added
        return len(self.list) == self.size and (not found)

    def __iter__(self):
        for elem in self.list:
            yield elem

    def __str__(self):
        return self.list.__str__()

    def __len__(self):
        return len(self.list)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_rewards(episode_rewards, show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


