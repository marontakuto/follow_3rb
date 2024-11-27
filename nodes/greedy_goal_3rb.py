# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger

import numpy as np

from pfrl import explorer


def select_action_epsilon_greedily(epsilon, random_action_func,
                                   greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class MyConstantEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with constant epsilon.
    Args:
      epsilon: epsilon used
      random_action_func: function with no argument that returns action
      logger: logger used
      replay_start_size: if t<replay_start_size, returns greedy action
    """

    def __init__(self, epsilon, random_action_func,
                 logger=getLogger(__name__), replay_start_size=0):
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.random_action_func = random_action_func
        self.logger = logger
        self.replay_start_size = replay_start_size

    def select_action(self, t, greedy_action_func, action_value=None):
        if t < self.replay_start_size:
            a, greedy = select_action_epsilon_greedily(
                1, self.random_action_func, greedy_action_func)
                
        else:
            a, greedy = select_action_epsilon_greedily(
                self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        return a

    def __repr__(self):
        return 'ConstantEpsilonGreedy(epsilon={})'.format(self.epsilon)


class MyLinearDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with linearyly decayed epsilon
    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func, logger=getLogger(__name__), replay_start_size=0):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon
        self.replay_start_size = replay_start_size

    def compute_epsilon(self, t):
        if t < self.replay_start_size:
            return 1
        elif t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon - epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        return a

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)

class MyEpsilonGreedy_old(explorer.Explorer):
    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func, logger=getLogger(__name__), replay_start_size=1000):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon
        self.replay_start_size = replay_start_size

    def compute_epsilon(self, t):
        if t > self.decay_steps + self.replay_start_size:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon - epsilon_diff * (t / self.decay_steps)

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = self.select_action_epsilon_greedily(self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        if t < self.replay_start_size:
            actionnum = np.random.rand()
            if actionnum < 0.5:
                a = 0
            elif actionnum >= 0.5 and actionnum < 0.75:
                a = 1
            elif actionnum >= 0.75:
                a = 2
            #else:
                #a = 3
        return a

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)

class MyEpsilonGreedy(explorer.Explorer): #2021年改良版
    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps, random_action_func, logger=getLogger(__name__), replay_start_size=1000, action_size=4):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon
        self.replay_start_size = replay_start_size
        self.action_size = action_size

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.start_epsilon - self.end_epsilon
            return self.start_epsilon - epsilon_diff * (t / (self.decay_steps))

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = self.select_action_epsilon_greedily(self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        if t < self.replay_start_size:
            a = np.random.randint(self.action_size) 
        return a

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)

class MyConstantGreedy(explorer.Explorer): #ConstantGreedy
    def __init__(self,  random_action_func, logger=getLogger(__name__), epsilon=0, replay_start_size=1000 ,action_size=3):
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.random_action_func = random_action_func
        self.logger = logger
        self.replay_start_size = replay_start_size
        self.action_size = action_size

    def select_action_epsilon_greedily(self, epsilon, random_action_func, greedy_action_func):
        if np.random.rand() < epsilon:
            return random_action_func(), False
        else:
            return greedy_action_func(), True

    def select_action(self, t, greedy_action_func, action_value=None):
        a, greedy = self.select_action_epsilon_greedily(self.epsilon, self.random_action_func, greedy_action_func)
        greedy_str = 'greedy' if greedy else 'non-greedy'
        self.logger.debug('t:%s a:%s %s', t, a, greedy_str)
        if t < self.replay_start_size:
            a = np.random.randint(self.action_size)
        return a

    def __repr__(self):
        return 'ConstantGreedy(epsilon={})'.format(self.epsilon)