# see https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa


import numpy as np
import gym
from gym import wrappers
import time
import pandas as pd


def persist_rewards(problem,episode, rewards, gamma):
    '''
     [episode, step_idx, total_reward, gamma]
    :param problem:
    :param rewards: list of form [episode, step_idx, total_reward, gamma]
    :param gamma:
    :return:
    '''
    data = np.array(rewards)
    df = pd.DataFrame(data, columns=['episode','iteration', 'reward', 'gamma'])
    df.to_csv(f"data/{problem.spec.id}/rewards_{episode}_stats_gamma{gamma}.csv")

def persist_stats(problem, stats, gamma, type):
    '''

    :param stats: the list of iteration frames
    :param gamma: the gamma used to calculate
    :param type: either 'value' or 'policy'
    :return: None
    '''
    df = pd.DataFrame(stats, columns=['iterations', 'delta_max', 'delta_mean', 'span', 'gamma'])
    df.to_csv(f"data/{problem.spec.id}/policy_iteration_{type}_stats_gamma{gamma}.csv")


# see hiivemdptoolbox
def get_span(array):
    """Return the span of `array`

    span(array) = max array(s) - min array(s)

    """
    return array.max() - array.min()

def calc_stats(iteration, previous_value_fn, value_fn, gamma):
    '''
    calc_stats
    :param iteration: the iteration number
    :param previous_value_fn: the prior value function
    :param value_fn:  the new value function
    :param gamma : current discount factor
    :return: [iteration, delta_max, delta_mean, gamma]
    '''

    the_diff = np.abs(previous_value_fn - value_fn)

    max_diff = np.max(the_diff)
    mean_diff = np.mean(the_diff)
    span = get_span(the_diff)

    return [iteration, max_diff, mean_diff, span,gamma]

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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def run_episode(env, episode,  policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    stats=[]
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        # TODO persist rewards here
        stats.append([episode, step_idx, total_reward, gamma])
        if done:
            break
    persist_rewards(env,episode,stats, gamma)
    return total_reward, step_idx


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    data = [run_episode(env, i, policy, gamma, False) for i in range(n)]
    scores = [item[0] for item in data]
    steps = [item[1] for item in data]
    return np.mean(scores), np.mean(steps)

@timing
def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


@timing
def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    num_states = env.observation_space.n
    v = np.zeros(num_states)
    eps = 1e-03
    while True:
        prev_v = np.copy(v)
        for s in range(num_states):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v



@timing
def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.random.choice(num_states, size=(num_actions))  # initialize a random policy
    max_iterations = 200000
    stats = []
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        # TODO score span, iteration, mean here
        stats.append(calc_stats(i, policy, new_policy, gamma))
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    persist_stats(env, stats, gamma, 'policy')
    return policy


if __name__ == '__main__':
    env_name  = 'Taxi-v3'
    gamma = 0.6
    env = gym.make(env_name)
    optimal_policy = policy_iteration(env, gamma)
    scores = evaluate_policy(env, optimal_policy, gamma)
    print('Average scores = ', np.mean(scores))