"""
Multi-armed Bandit
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
    def __init__(self, name, actions=None):
        self.name = name
        self.actions = actions
        self.n_actions = len(actions)
        self.q_values = np.zeros(self.n_actions)
        self.imps = np.zeros(self.n_actions)
        self.total_reward = 0
        self.avg_reward = list()

    # ??? can be used 2 methods only: for e-greedy and ucb, is it worth it?
    def update(self, a_idx, reward):
        """
        update q_values of selected action with given reward following formula:
        Q_n+1 = Q_n + (1/n)(R_n+Q_n)

        NOTE: this update allows only to three methods: ABn, eGreedy, and UCB
        """
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
    def __init__(self, name, actions):
        super().__init__(name, actions)

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

    def run_prod(self, n_prod):
        """
        run n_prod times with optimal action from achieved from run_test.
        """
        for i in range(n_prod):
            reward = self.best_a.display_frame()
            self.total_reward += reward
            avg_reward_so_far = self.total_reward/(i+1)
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
    def __init__(self, name, actions, eps):
        super().__init__(name, actions)
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
            self.update(a_idx, reward)


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
    def __init__(self, name, actions, c):
        super().__init__(name, actions)
        self.c = c
        self.act_indices = np.array(range(self.n_actions))

    def run(self, n_prod):
        for i in range(n_prod):
            if any(self.imps == 0):# randomly choose from frames with NO impressions
                a_idx = np.random.choice(self.act_indices[self.imps==0])
            else:
                uncertainty = np.sqrt(np.log(i+1) / self.imps)
                a_idx = np.argmax(self.q_values + self.c * uncertainty)

            # ??? all methods(abn, egreedy, ucd) below is redundant for all methods -> must refactor
            # add diff run for thompson only, it will make code much more simpler!!!
            reward = self.actions[a_idx].display_frame()
            self.update(a_idx, reward)




# ??? whether to write run function for each method classes
# or just one(has some bottlenecks as well)?
def run(mab_method, n_prod):
    """
    Run production ready multi-armed bandit comparison methods for n_prod times.

    Parameters
    ----------
    mab_method: object
            Instantiated object of class inherited from ComparativeMethodsMAB.

    n_prod: int
            number of iteration to be ran.

    """



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
