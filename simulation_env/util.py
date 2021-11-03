"""
utility functions for simulation environments

Development legend(should be deleted once done):
  *** = todos,
  ??? = fixes
"""

from simulation_env.advertisement import Advertisement

def rand_update_ads(ad_lst, lb, ub):
    """
    Update each Ad in ad_lst randomly with given bound.

    ad_lst : list of Advertisement objects
    lb,up : float
         lower and upper bound for update percentage
         ex: if lb,ub= 0.05
         ctr update range is [0.95, 1.05]
    """
    for ad in ad_lst:
        ad.rand_update(lb, ub)