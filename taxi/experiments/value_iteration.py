
import numpy as np
import gym
from gym import wrappers


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = True)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

def evaluate_rewards_and_transitions(problem, mutate=False):
    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T

def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
  # initialize value-function
    max_iterations = 100000
    eps = 10**-3

    value_fn = np.zeros(env.observation_space.n)
    R, T = evaluate_rewards_and_transitions(env)
    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)

        if np.max(np.abs(value_fn - previous_value_fn)) < eps:
            break
    return value_fn


if __name__ == '__main__':
    env_name  = 'Taxi-v3'
    gamma = 0.6
    env = gym.make(env_name)
    optimal_v = value_iteration(env, gamma);
    policy = extract_policy(env, optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)
