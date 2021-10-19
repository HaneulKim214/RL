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
    def __init__(self, ctr, name, alpha, beta):
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

        self.alpha = alpha
        self.beta = beta

    def display_ad(self):
        """
        Display advertisement of choice and return reward.
        reward = 1 if clicked else 0. which is chosen from
        binomial distribution with given probability p.
        """
        reward = np.random.binomial(n=1, p=self.ctr)
        self.imps += 1
        self.rewards += reward
        self.alpha += reward
        self.beta += 1 - reward
        return reward

    def reset(self):
        """
        reset number of impressions and rewards
        """
        self.alpha = 1
        self.beta = 1 if self.beta != np.inf else np.inf

    def __str__(self):
        return f'{self.name}'


def check_ctr(ctr):
    if (ctr == float("inf")) or (np.isnan(ctr)):
        return 0
    else:
        return ctr