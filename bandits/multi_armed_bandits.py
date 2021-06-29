"""
Multi-armed Bandit


Development legend(should be deleted once done):
  *** = todos,
  ??? = fixes
"""

# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import pandas as pd
import numpy as np

from frame import Frame



class ComparativeMethodsMAB:
    """
    Class containing variety of comparative methods for MAB problems

    Each method requires name(str), actions(list of frame objects),
    """
    def __init__(self, actions=None):
        self.actions = actions
        self.n_actions = len(actions)
        self.q_values = np.zeros(self.n_actions)
        self.imps = np.zeros(self.n_actions)
        self.total_reward = 0
        self.avg_reward = list()

    def update(self, a_idx, reward, i):
        """
        Update function
        formula for ABn, eGreedy, and UCB:
            Q_n+1 = Q_n + (1/n)(R_n+Q_n)

        formula for ThompsonSampling:
            A_k += R_t, B_k += 1 - R_t

        Parameters
        ----------
        a_idx : int
                index of selected action, it is used to get action from list of actions.

        reward : int
                 reward agent received at each episode

        i : int
            number of episode
        """
        # if self.name not in ["ABn", "eGreedy", "UCB"]:
        #     raise ValueError("ComparativeMethodsMAB update method "
        #                      "supports only ABn, eGreedy, and UCB")

        if self.name == "ThompsonSampling":
            self.alphas[a_idx] += reward
            self.betas[a_idx] += 1 - reward

        elif self.name in ["ABn", "eGreedy", "UCB"]:
            self.imps[a_idx] += 1
            self.q_values[a_idx] += (1 / self.imps[a_idx]) * (reward - self.q_values[a_idx])

        self.total_reward += reward
        avg_reward_so_far = self.total_reward / (i + 1)
        self.avg_reward.append(avg_reward_so_far)

    @staticmethod
    def test1():
        print("hello word2")

    @property # this allow below code to be used like an attribute
    def describe_(self):
        return f'using {self.name} method with actions = {self.actions}, q_values = {self.q_values}'

    def __repr__(self):
        return f"ComparativeMethodsMAB({self.name}, {self.actions})"

    def __str__(self):
        return self.name


class ABn(ComparativeMethodsMAB):
    def __init__(self, actions):
        super().__init__(actions)
        self.name = "ABn"

    def run_test(self, n_test):
        """
        Run test experiment for n_test times to choose optimal actions

        update using following formula:
        Q_n+1 = Q_n + (1/n)(R_n+Q_n)
        """
        for i in range(n_test):
            a_idx = np.random.randint(self.n_actions)
            reward = self.actions[a_idx].display_frame()
            self.update(a_idx, reward, i)

        self.best_a = self.actions[np.argmax(self.q_values)]
        self.n_test = n_test

    def run_prod(self, n_prod):
        """
        run n_prod times with optimal action from achieved from run_test.
        """
        for i in range(n_prod):
            reward = self.best_a.display_frame()
            self.total_reward += reward
            avg_reward_so_far = self.total_reward/(self.n_test+i+1)
            self.avg_reward.append(avg_reward_so_far)


class eGreedy(ComparativeMethodsMAB):
    """
    method that choose random action with probability of eps value else choose
    optimal actions.

    Q-update function is same as A/B/n test however two main differences
     1. one additional hyperparameter epsilon
     2. Choose action with highest q_value from the start

    Parameters
    ----------
    eps : float
         between (0,1], greater eps => greater exploration meaning chooses action randomly
         even though it might not be optimal.

    """
    def __init__(self, actions, eps):
        super().__init__(actions)
        self.name = "eGreedy"
        self.eps = eps

    def run(self, n_prod):
        """
        choose best actions mostly with eps% of randomly choosing an action.
        update q_values appropriately.

        Parameters
        ---------
        n_prod : int
                 number of iteration to be ran.
        """
        for i in range(n_prod):
            if np.random.uniform() <= self.eps or i == 0:
                a_idx = np.random.randint(self.n_actions)
            else:
                a_idx = np.argmax(self.q_values)
            reward = self.actions[a_idx].display_frame() # ??? this should be more general since rn it is
                                                         # specialized to frame object only.
            self.update(a_idx, reward, i)


class UpperConfidenceBounds(ComparativeMethodsMAB):
    """
    Methods that resolve exploration-exploitation trade-off even in non-stationary environment.
    best action is selected based on its potential for reward which is calculated as
    sum of action value estimate and measure of uncertainty of this estimate.

    Q-update function is same as A/B/n test however one main differences
     1. one additional hyperparameter c

    Parameters
    ----------
    c : float
       hyperparameter for uncertainty measure

    """
    def __init__(self, actions, c):
        super().__init__(actions)
        self.name = "UCB"
        self.c = c
        self.act_indices = np.array(range(self.n_actions))

    def run(self, n_prod):
        for i in range(n_prod):
            if any(self.imps == 0):# randomly choose from frames with NO impressions
                a_idx = np.random.choice(self.act_indices[self.imps==0])
            else:
                uncertainty = np.sqrt(np.log(i+1) / self.imps)
                a_idx = np.argmax(self.q_values + self.c * uncertainty)
            reward = self.actions[a_idx].display_frame()
            self.update(a_idx, reward, i)


class ThompsonSampling(ComparativeMethodsMAB):
    """
    Methods that resolve exploration-exploitation trade-off even in non-stationary environment.
    best action is selected based on its potential for reward which is calculated as
    sum of action value estimate and measure of uncertainty of this estimate.

    Q-update function is same as A/B/n test however one main differences
     1. one additional hyperparameter c

    Parameters
    ----------
    c : float
       hyperparameter for uncertainty measure

    """
    def __init__(self, actions):
        super().__init__(actions)
        self.name = "ThompsonSampling"

        # parameters in beta distribution, initialized to 1 however could use previous data to initialize.
        self.alphas = np.ones(self.n_actions)
        self.betas = np.ones(self.n_actions)

    def run(self, n_prod):
        for i in range(n_prod):
            theta_samples = [np.random.beta(self.alphas[k], self.betas[k]) \
                             for k in range(self.n_actions)]
            a_idx = np.argmax(theta_samples)
            reward = self.actions[a_idx].display_frame()
            self.update(a_idx, reward, i)




"""
------------- just some notes for myself -----------
Builtin function 
- isinstance() -> to check if object belong to specific class. ex: print(isinstance(a, ABn))
  returns True/False
- issubclss() -> check if one class if subclass of another. ex: print(issubclass(ABn, ComparativeMethodsMAB))
  return True/False
-------------
Others
- never pass mutable datatype as default arguments. ex: don't do c=[]

Magic/Dunder methods
__repr__
repr(ABn)
- unambiguous represenation of the object, used for debugging, logging, etc... for developers
- this gets returned upon print statement when __str__ is not present.

__str__
str(ABn)
- readable representation of object, for end users


hotkeys
- multiple caret drag: shift+alt drag
- duplicate code below: ctrl + d
- move line up/down :  shift+alt up/down

"""
