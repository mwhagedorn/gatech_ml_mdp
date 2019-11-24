import gym
from gym import envs
print(envs.registry.all())

env_name = 'NChain-v0'
# gamma = 0.8
env = gym.make(env_name).env

num_states = env.observation_space.n
num_actions = env.action_space.n

print("num states", num_states, "num actions", num_actions)