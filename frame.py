"""
Frames
"""

# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import pandas as pd
import numpy as np

class Frame:
    """
    Instantiate Frame object either combi or stationary with their CTR
    """
    def __init__(self, ctr, name):
        self.ctr = ctr # ctr can we replaced with diff wilson_ctr
        self.name = name

    def display_frame(self):
        """
        Display frame of choice and return reward.
        reward = 1 if clicked else 0. which is chosen from
        binomial distribution with given probability p.

        """
        reward = np.random.binomial(n=1, p=self.ctr)
        return reward

    def __str__(self):
        return f'{self.name}'