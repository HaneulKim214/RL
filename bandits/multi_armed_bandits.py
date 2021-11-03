"""
Multi-armed Bandit problem's exploration exploitation strategies

Development legend(should be deleted once done):
  *** = todos,
  ??? = questions
"""
# Author: Haneul Kim <haneulkim214@gmail.com>


import pandas as pd
import numpy as np

from simulation_env.advertisement import Advertisement


class ComparativeMethodsMAB:
    """
    Class containing variety of comparative methods for MAB problems

    Each method requires name(str), actions(list of frame objects),
    """
    def __init__(self, batch_size_i, actions):
        self.actions = actions
        self.n_actions = len(actions)
        self.q_values = np.zeros(self.n_actions)
        self.imps = np.zeros(self.n_actions)

        self.rewards = []
        self.a_idxes = []
        self.regret = []

    def calc_avg_reward(self):
        cum_tot_rewards = np.cumsum(self.rewards)
        self.avg_reward = cum_tot_rewards / (np.arange(1, len(self.rewards) + 1))

    def update(self, a_idx, reward):
        """
        Update action value
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
        self.rewards.append(reward)

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
    def __init__(self, batch_size_i, actions):
        super().__init__(batch_size_i, actions)
        self.name = "ABn"

    def run_test(self, n_test):
        """
        Run test experiment for n_test times to choose optimal actions

        update using following formula:
        Q_n+1 = Q_n + (1/n)(R_n+Q_n)
        """
        for i in range(n_test):
            a_idx = np.random.randint(self.n_actions)
            reward = self.actions[a_idx].display_ad()
            self.update(a_idx, reward)

        self.best_a = self.actions[np.argmax(self.q_values)]
        self.n_test = n_test

    def run_prod(self, n_prod):
        """
        run n_prod times with optimal action from achieved from run_test.
        """
        for i in range(n_prod):
            reward = self.best_a.display_ad()
            self.total_reward += reward
            avg_reward_so_far = self.total_reward/(self.n_test+i+1)
            self.avg_reward.append(avg_reward_so_far)

    def run_8_logic(self, n_prod):
        """
        Run simulation using logic #8
        """
        best_action = sorted(self.actions, key=lambda x: x.ctr, reverse=True)[0]
        desc_actions = sorted(self.actions, key=lambda x: x.pred_ctr, reverse=True)
        gun1, gun2 = desc_actions[0], desc_actions[1:]

        for i in range(1, n_prod+1):
            # 5% 확률로 2군
            if np.random.uniform() <= 0.05:
                frame = np.random.choice(gun2)
            else:
                frame = gun1
            reward = frame.display_ad()
            self.regret.append(best_action.ctr - frame.ctr)

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
    def __init__(self, batch_size_i, actions, eps):
        super().__init__(batch_size_i, actions)
        self.name = "eGreedy"
        self.eps = eps

    def run_test(self, n_prod):
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
            reward = self.actions[a_idx].display_ad()
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
    def __init__(self, batch_size_i, actions, c):
        super().__init__(batch_size_i, actions)
        self.name = "UCB"
        self.c = c
        self.act_indices = np.array(range(self.n_actions))

    def run_test(self, n_prod):
        for i in range(n_prod):
            if any(self.imps == 0):# randomly choose from frames with NO impressions
                a_idx = np.random.choice(self.act_indices[self.imps==0])
            else:
                uncertainty = np.sqrt(np.log(i+1) / self.imps)
                a_idx = np.argmax(self.q_values + self.c * uncertainty)
            reward = self.actions[a_idx].display_ad()
            self.update(a_idx, reward)


class ThompsonSampling(ComparativeMethodsMAB):
    """
    Methods that resolve exploration-exploitation trade-off even in non-stationary
    environment. Best action is selected based on its potential for reward
    which is calculated assum of action value estimate and measure of
    uncertainty of this estimate.

    Q-update function is same as A/B/n test however one main differences
     1. one additional hyperparameter c

    Parameters
    ----------
    batch_size_i : int
                 update alpha, beta value every batch_size_i time
    actions : list
              each element is possible Advertisement

    initialize : list
    """
    def __init__(self, batch_size_i, actions):
        super().__init__(batch_size_i, actions)
        self.name = "ThompsonSampling"

        # *** parameters in beta distribution, initialized to 1 however could
        # use previous data to initialize.
        self.alphas = np.array([act.alpha for act in actions])
        self.betas = np.array([act.beta for act in actions])

    def run_test(self, n_prod, batch_size_i=np.inf):
        """
        Parameters
        ---------
        batch_size_i : int
                     for batch_update within a day
                     default = +infinity => never update within run_test duration
        """
        for i in range(1, n_prod+1):
            q_vals = [np.random.beta(self.alphas[k], self.betas[k]) \
                             for k in range(self.n_actions)]
            a_idx = np.argmax(q_vals)
            # *** In production display_ad() should be user's interaction!
            reward = self.actions[a_idx].display_ad()
            self.imps[a_idx] += 1
            self.rewards.append(reward)
            self.a_idxes.append(a_idx)

            if i % batch_size_i == 0:
                self.update_batch()

    def reset_params(self):
        """
        Reset parameters alpha, beta
        """
        # reset alpha, beta values only since other need to be saved for analysis
        self.alphas = np.ones(self.n_actions)
        self.betas = np.array([act.beta for act in self.actions])
        self.q_values = np.zeros(self.n_actions)

    def run(self):
        """
        For *Production*
           everytime new slot for ad opens up it selects value of each action from
           its beta distribution then frame with largest value gets chosen
           Thompson sampling applied to production setting where input/reward
           will be given from JAVA, reward is actual user click/no_click
        """
        # ??? Is it okay to only random sample from selected actions?
        q_vals = [np.random.beta(self.alphas[k], self.betas[k]) \
                         for k in range(self.n_actions)]
        a_idx = np.argmax(q_vals)
        frame = self.actions[a_idx]
        return a_idx, frame

    def update_batch(self):
        """Update alpha, beta when using data collected until
           this method is called then reset it for next batch.
        """
        for a_idx, reward in zip(self.a_idxes, self.rewards):
            self.update(a_idx, reward)
        self.rewards = list()
        self.a_idxes = list()


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
