"""
Contextual Bandit

MAB 와 같이 하나의 Bandit 으로 관리할수 있지만
1. Simple 한 MAB 적용
2. CB 적용
순서로 할거이기 때문에 1번 production 에 올라가있을때 2번 develop
하려면 따로 관리하는게 나을것 같음.

Development legend(should be deleted once done):
  *** = todos,
  ??? = fixes
"""

# Author: Haneul Kim <haneulkim214@gmail.com>
# License: Enliple

import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy import stats
import sys


def select_ad_thompson(ad_models, context):
    samples = {ad: ad_models[ad].sample_prediction(context) for ad in ad_models}
    max_value = max(samples.values())

    # Tie breaker
    max_keys = [key for key, value in samples.items() if value == max_value]
    return np.random.choice(max_keys)


class RegularizedLR:
    def __init__(self, name, alpha, rlambda, n_dim):
        self.name = name
        self.alpha = alpha
        self.rlambda = rlambda
        self.n_dim = n_dim

        self.m = np.zeros(n_dim)  # mean estimate of w[i] is given by m[i], variance estimat = inv(q[i])
        self.q = np.ones(n_dim) * rlambda

        self.w = self.get_sampled_weights()  # weights of paremters in logistic

    def get_sampled_weights(self):
        """
        Samples parameters of logistic regression
        """

        w = np.random.normal(self.m, self.alpha * self.q ** (-1 / 2))
        return w

    def loss(self, w, *args):
        """For training"""

        X, y = args
        n = len(y)
        regularizer = 0.5 * np.dot(self.q, (w - self.m) ** 2)

        pred_loss = sum([np.log(1 + np.exp(np.dot(w, X[j])))
                         - y[j] * np.dot(w, X[j]) for j in range(n)])
        return regularizer + pred_loss

    def fit(self, X, y):
        if y:
            X = np.array(X)
            y = np.array(y)
            minimization = minimize(self.loss, self.w, args=(X, y),
                                    method="L-BFGS-B", bounds=[(-10, 10)] * 3 + [(-1, 1)],
                                    options={'maxiter': 50})
            self.w = minimization.x
            self.m = self.w  # prev weights serve as prior
            p = (1 + np.exp(-np.matmul(self.w, X.T))) ** (-1)
            self.q = self.q + np.matmul(p * (1 - p), X ** 2)

    def calc_sigmoid(self, w, context):
        return 1 / (1 + np.exp(-np.dot(w, context)))

    def get_prediction(self, context):
        return self.calc_sigmoid(self.m, context)

    def sample_prediction(self, context):
        w = self.get_sampled_weights()
        return self.calc_sigmoid(w, context)

    def get_ucb(self, context):
        pred = self.calc_sigmoid(self.m, context)
        confidence = self.alpha * np.sqrt(np.sum(np.divide(np.array(context) ** 2, self.q)))
        ucb = pred + confidence
        return ucb


class UserGenerator(object):
    def __init__(self):
        self.beta = {}

        # weights for [bias, location, device, age] for each Advertisement A, B, C, D, E...
        self.beta['A'] = np.array([-4, -0.1, -3, 0.1])
        self.beta['B'] = np.array([-6, -0.1, 1, 0.1])
        self.beta['C'] = np.array([2, 0.1, 1, -0.1])
        self.beta['D'] = np.array([4, 0.1, -3, -0.2])
        self.beta['E'] = np.array([-0.1, 0, 0.5, -0.01])
        self.context = None

    def logistic(self, beta, context):
        """
        Given user context, return click/no click
        """
        f = np.dot(beta, context)
        p = 1 / (1 + np.exp(-f))
        return p

    def display_Advertisement(self, frame):
        p = self.logistic(self.beta[ad], self.context)  # % of click by using info of context
        reward = np.random.binomial(n=1, p=p)
        return reward

    def generate_user_with_context(self):
        # 0: International, 1: U.S.
        location = np.random.binomial(n=1, p=0.6)
        # 0: Desktop, 1: Mobile
        device = np.random.binomial(n=1, p=0.8)
        # User age changes between 10 and 70,
        # with mean age 34
        age = 10 + int(np.random.beta(2, 3) * 60)
        # Add 1 to the concept for the intercept
        self.context = [1, device, location, age]
        return self.context