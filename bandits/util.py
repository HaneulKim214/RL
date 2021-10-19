import numpy as np


def create_context():
    location = np.random.binomial(n=1, p=0.6)
    # 0: Desktop, 1: Mobile
    device = np.random.binomial(n=1, p=0.8)
    # User age changes between 10 and 70,
    # with mean age 34
    age = 10 + int(np.random.beta(2, 3) * 60)
    # Add 1 to the concept for the intercept
    context = [1, device, location, age]
    return context

def select_ad_thompson(frames, context):
    """
    Return ad with highest expectation of click given context

    compute lr for each ad using context and pick highest value => highest % of click
    """
    q_val = [frame.model.sample_prediction(context) for frame in frames]
    # *** Need to add tie breaker, else select first largest q_val
    # ??? is it necessary?
    return np.argmax(q_val)

def logistic(weights, context):
    f = np.dot(weights, context)
    p = 1 / (1 + np.exp(-f))
    return p

def display_frame(weights, context):
    """
    Simulation of user click
    """
    p = logistic(weights, context) # % of click by using info of context
    reward = np.random.binomial(n=1, p=p)
    return reward

def calc_regret(frames, context, ad_idx):
    action_vals = [logistic(frame.weights, context) for frame in frames]
    best_a_idx = np.argmax(action_vals)

    regret = action_vals[best_a_idx] - action_vals[ad_idx]
    return regret, best_a_idx
