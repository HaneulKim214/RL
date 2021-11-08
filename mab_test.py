from datetime import datetime, timedelta
import itertools
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import sys

# In Github
from bandits.multi_armed_bandits import *
from simulation_env.advertisement import Advertisement
from simulation_env.util import rand_update_ads

batch_size_i = 1
ad_A = Advertisement(0.115, "Ad_A")
ad_C = Advertisement(0.103, "Ad_C")
ads = [ad_A, ad_C]

n_exp = 1000
ts_alg = ThompsonSampling(1, ads)
ts_alg.run_test(n_exp)

for day_i in range(10):
    ts_alg.run_test(n_exp)
    ad_A.rand_update(0.03, 0.01) # ctr decrease as time pass
    ad_C.rand_update(0.01, 0.03)