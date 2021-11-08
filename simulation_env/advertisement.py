"""
Advertisement simulation environment

Development legend(should be deleted once done):
  *** = todos,
  ??? = fixes
"""

# Author: Haneul Kim <haneulkim214@gmail.com>

from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class Advertisement:
    """
    Instantiate Advertisement object
    """
    def __init__(self, ctr, name):
        """
        Parameters
        ----------
        imps : int
              number of times frame has been sent out
        """
        self.ctr = check_ctr(ctr)
        self.pred_ctr = 0
        self.name = name
        self.imps = 0
        self.rewards = 0
        self.ctr_history = []
        self.ctr_history.append(self.ctr)

    def add_ad(self):
        """
        New advertisement added to simluation env
        """

    def display_ad(self):
        """
        Display advertisement of choice and return reward.
        reward = 1 if clicked else 0. which is chosen from
        binomial distribution with given probability p.
        """
        reward = np.random.binomial(n=1, p=self.ctr)
        self.imps += 1
        self.rewards += reward
        return reward

    def rand_update(self, lb, ub):
        """
        Updates CTR randomly.
        To mimic non-stationary reward distribution of real-world

        lb,up : float
             lower and upper bound for update percentage
             ex: if lb,ub= 0.05
             ctr update range is [0.95, 1.05]
        """
        update_p = np.random.uniform(low=-lb, high=ub)
        self.ctr = self.ctr * (1+update_p)
        self.ctr_history.append(self.ctr)

    def reset_imps(self):
        self.imps = 0

    def __str__(self):
        return f'{self.name}'


def check_ctr(ctr):
    if (ctr == float("inf")) or (np.isnan(ctr)):
        return 0
    else:
        return ctr