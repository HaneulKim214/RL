import pandas as pd
import pickle
import numpy as np

from multi_armed_bandits import *
from contextual_bandits import *

from simulation_env.advertisement import Advertisement

alpha = 1
beta = 1
Ad1 = Advertisement(0.004, "Ad1", alpha, beta)
Ad2 = Advertisement(0.016, "Ad2", alpha, beta)
Ad3 = Advertisement(0.02, "Ad3", alpha, beta)
Ad4 = Advertisement(0.028, "Ad4", alpha, beta)
Ad5 = Advertisement(0.031, "Ad5", alpha, beta)
Ads = [Ad1, Ad2, Ad3, Ad4, Ad5]

# ABn
batch_size = 100
abn_method = ABn(batch_size, Ads)
print(abn_method.describe_)
abn_method.run_test(n_test=10000)
print(f"best frame = {abn_method.actions[np.argmax(abn_method.q_values)]}")
print(f"{abn_method.best_a}")
abn_method.run_prod(n_prod=50000)

# eGreedy
n_prod = 60000
eps = 0.1
egreedy = eGreedy(batch_size, Ads, eps)
egreedy.run_test(n_prod=n_prod)

# UCB
c = 0.1
ucb = UpperConfidenceBounds(batch_size, Ads, c)
ucb.run_test(n_prod=n_prod)

# Thompson Sampling
ts = ThompsonSampling(batch_size, Ads)
ts.run_test(n_prod=n_prod)


# Summary
print()
print("----------------")
print("Total rewards")
print(f"A/B/n methods = {abn_method.total_reward}")
print(f"e-greedy methods = {egreedy.total_reward}")
print(f"UCB methods = {ucb.total_reward}")
print(f"ThompsonSampling methods = {ts.total_reward}")