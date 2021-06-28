import pandas as pd
import numpy as np

from mab import ABn, eGreedy, UpperConfidenceBounds
from frame import Frame

Frame1 = Frame(0.004, "frme1")
Frame2 = Frame(0.016, "frme2")
Frame3 = Frame(0.02, "frme3")
Frame4 = Frame(0.028, "frme4")
Frame5 = Frame(0.031, "frme5")
frames = [Frame1, Frame2, Frame3, Frame4, Frame5]

# ABn
abn_method = ABn("A/B/n method", frames)
print(abn_method.describe_)
abn_method.run_test(n_test=10000)
print(f"best frame = {abn_method.actions[np.argmax(abn_method.q_values)]}")
print(f"{abn_method.best_a}")
abn_method.run_prod(n_prod=50000)

# eGreedy
n_prod = 60000
eps = 0.1
egreedy = eGreedy("e-greedy", frames, eps)
egreedy.run(n_prod=n_prod)

# UCB
c = 0.1
ucb = UpperConfidenceBounds("UCB", frames, c)
ucb.run(n_prod=n_prod)


# Summary
print()
print("----------------")
print("Total rewards")
print(f"A/B/n methods = {abn_method.total_reward}")
print(f"e-greedy methods = {egreedy.total_reward}")
print(f"UCB methods = {ucb.total_reward}")
