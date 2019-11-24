import numpy as np
import gym
from gym import wrappers
import time
import pandas as pd



def append_ql_episode_training_data(episode, penalty, alpha, gamma, epsilon):
    data.append([episode, penalty, alpha, gamma, epsilon])

def persist_ql_episodes_summary_data(problem, alpha, gamma, epsilon):
    df = pd.DataFrame(data, columns=['episode', 'penalty', 'alpha', 'gamma', 'epsilon'] )
    df.to_csv(f"data/{problem.spec.id}/ql_training_gamma{gamma}_{alpha}_{epsilon}.csv")


# Initialize the Q-table by all zeros.
# Start exploring actions: For each state, select any one among all possible actions for the current state (S).
# Travel to the next state (S') as a result of that action (a).
# For all possible actions from the state (S') select the one with the highest Q-value.
# Update Q-table values using the equation.
# Set the next state as the current state.
# If goal state is reached, then end and repeat the process.

data = []

def q_learning(env, alpha=0.7,gamma=0.6, epsilon=1.0 , max_steps = 100, total_episodes = 20000):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    max_epsilon = epsilon # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.01

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for episode in range(total_episodes):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        for step in range(max_steps):
            choice = np.random.uniform(0,1)
            if choice < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

            all_penalties.append(penalties)

        if episode % 100 == 0:
            print(f"Episode: {episode}  Avg penalties: {np.sum(all_penalties)/(episode+1)}")
        append_ql_episode_training_data(episode, penalties, alpha, gamma, epsilon)
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    persist_ql_episodes_summary_data(env, alpha, gamma, epsilon)
    print("Training finished.\n")
    return q_table

def evaluate_q_learner(env, qtable):
    """Evaluate agent's performance after Q-learning"""

    # α: (the learning rate) should decrease as you continue to gain
    #   a larger and larger knowledge base.
    # γ: as you get closer and closer to the deadline, your preference
    #   for near - term reward should increase, as you won't be around
    #   long enough to get the long-term reward, which means your gamma should decrease.
    # ϵ: as we develop our strategy, we have less need of exploration and more exploitation
    #    to get more utility from our policy, so as trials increase, epsilon should
    #    decrease.

    total_epochs, total_penalties = 0, 0
    episodes = 100
    total_rewards =[]
    gamma = 0.7
    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False
        total_reward = 0
        step_idx = 0
        disc_reward = 0
        while not done:
            action = np.argmax(qtable[state])
            state, reward, done, info = env.step(action)

            disc_reward += (gamma ** step_idx * reward)
            step_idx += 1

            if reward == -10:
                penalties += 1

            epochs += 1
            total_reward+=disc_reward

        total_penalties += penalties
        total_epochs += epochs
        total_rewards.append(total_reward)

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    print(f"Average rewards per episode: {np.mean(total_rewards)}")
