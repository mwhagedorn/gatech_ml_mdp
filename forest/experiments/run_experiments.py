import gym
from gym import wrappers
import time
import pandas as pd

import matplotlib.pyplot as plt
from forest.experiments.mdptoolbox import mdp, example

import pandas as pd







def value_iteration(P, R):
    gammas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]

    for gamma in gammas:
        pi = mdp.ValueIteration(P, R, gamma)
        pi.setVerbose()
        pi.run()

        print(pi.time)
        df = pd.DataFrame(pi.run_stats, columns=['State', 'Action', 'Max V', 'Error', 'Time', 'Value', 'Policy'])
        df.to_csv(f"data/forest_value_iteration_{gamma}_run_stats.csv")


def policy_iteration(P, R):
    gammas = [ 0.7]

    for gamma in gammas:
        pi = mdp.ValueIteration(P, R, gamma)
        pi.setVerbose()
        pi.run()
        print(pi.V)

        print(pi.time)
        df = pd.DataFrame(pi.run_stats, columns=['State', 'Action', 'Max V', 'Error', 'Time', 'Value', 'Policy'])
        df.to_csv(f"data/forest_policy_iteration_{gamma}_run_stats.csv")

        # P, R = get_transition_and_reward_arrays(0.5)
        # sdp = mdp.FiniteHorizon(P, R, 0.96, 50)
        # sdp.run()
        # return sdp
import numpy as np
np.random.seed(0)
def q_learning(P, R):
   gamma = 0.9
   ql =  mdp.QLearning(P, R, gamma)
   ql.run()
   print(len(ql.mean_discrepancy))
   df = pd.DataFrame(ql.run_stats, columns=['State', 'Action', 'Reward', 'Error', 'Time', 'Value', 'Alpha', 'Episilon', 'Max V', 'Value', 'Policy'])
   df.to_csv(f"data/forest_ql_{gamma}_run_stats.csv")
   errors = df['Error'].rolling(1000).mean()
   errors.to_csv(f"data/forest_ql_rolling_errs_{gamma}.csv")

from random import Random

def simulate_ideal_forest():

    runs = []
    for trial in range(100):
        idf = IdealForestAgent()
        for i in range(100):
            pn = Random().random()
            print(pn)
            # probability of fire
            if pn < 0.9:
                idf.step()
            else:
                idf.reset()

        runs.append(idf.succesful_runs)

    print(np.mean(runs))

class IdealForestAgent():
    def __init__(self):
        self.state=0
        self.succesful_runs=0

    def step(self):
        self.state+=1
        if self.state > 4:
            self.succesful_runs+=1
            self.state = 0

    def reset(self):
        self.state = 0

# Initialise a finite horizon MDP.


def run_forest():
    P, R = example.forest(S=5)
    #value_iteration(P, R)
    #policy_iteration(P, R)
    q_learning(P,R)
    simulate_ideal_forest()

run_forest()


