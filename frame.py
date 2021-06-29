"""
Frames

Development legend(should be deleted once done):
  *** = todos,
  ??? = fixes
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
        # ***dev Todos:
        # - add various ctr calculation methods such as wilson_ctr
        # - ctr is non-stationary, so must be updated which it does in mab however not here
        self.ctr = ctr
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