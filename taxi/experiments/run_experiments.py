import numpy as np
import gym
from gym import wrappers
import time
import pandas as pd

from taxi.experiments.policy_iteration import policy_iteration
from taxi.experiments.policy_iteration import evaluate_policy
from taxi.experiments.policy_iteration import get_span

from  taxi.experiments.value_iteration import value_iteration
from  taxi.experiments.value_iteration import extract_policy

from taxi.experiments.q_learning import q_learning
from taxi.experiments.q_learning import evaluate_q_learner

def run_taxi():
    env_name = 'Taxi-v3'
    gamma = 0.8
    env = gym.make(env_name)

    optimal_policy = policy_iteration(env, gamma)
    mean_score, mean_steps= evaluate_policy(env, optimal_policy, gamma, n=100)
    print('(Policy)Average scores = ', np.mean(mean_score))
    print('(Policy)Num Steps scores = ', np.mean(mean_steps))

    optimal_v = value_iteration(env, gamma);
    policy = extract_policy(env,optimal_v, gamma)
    policy_score, steps = evaluate_policy(env, policy, gamma, n=100)
    print('(Value) Policy average score = ',policy_score)
    print('(Value) Policy average steps = ', steps)
    span = get_span(np.fabs(optimal_policy - policy))
    print(f"Policy Difference (span) {span} ")

    qtable = q_learning(env, alpha=0.8)
    evaluate_q_learner(env, qtable)




run_taxi()


# policy_iteration function took 1764.131 ms
# (Policy)Average scores =  -2.8327845283404183
# (Policy)Num Steps scores =  12.72
# (Value) Policy average score =  -2.649468085449523
# (Value) Policy average steps =  12.32
# Policy Difference (span) 0.0
