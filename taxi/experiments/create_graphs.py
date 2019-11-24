import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# If Gamma is closer to one, the agent will consider future rewards with greater weight, ...
def create_graphs():
    gammas = [ 0.7]
    alphas = [0.4, 0.5, 0.6, 0.7]
    dfs = []
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')

    def policy_iteration():
        for gamma in gammas:
            df = pd.read_csv(f"/Users/mwhagedorn/develop/gatech/ml/mdp/taxi/experiments/data/Taxi-v3/policy_iteration_policy_stats_gamma{gamma}.csv")
            df['gamma'] = gamma
            dfs.append(df)

        ds = 0
        legends= []
        for data in dfs:

            iterations = data[data.columns[0]]
            error = data['span']
            gamma = data['gamma'][0]

            plt.plot(iterations,error)
            legends.append(f"gamma={gamma}")

        plt.title("Taxi: policy iteration error(span) vs iterations")
        plt.legend(legends, loc='upper right')
        plt.show()


        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            max_v = data[data.columns[3]]
            gamma = data['gamma']

            plt.plot(iterations, max_v, label="cumulative_reward")
            legends.append(f"gamma={gamma}")

        plt.title("Taxi: policy iteration max reward vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()

        legends = []
        # for data in dfs:
        #     iterations = data[data.columns[0]]
        #     time = data['Time']
        #     gamma = data[data.columns[8]][0]
        #     running_time = time.cumsum()
        #
        #     plt.plot(iterations,running_time,  label="cumulative_time")
        #     legends.append(f"gamma={gamma}")

        plt.title("Taxi: policy iteration cumulative time vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()

        # collect episode info
        rewards = []
        steps = []
        for index in range(99):
            df = pd.read_csv(
                f"/Users/mwhagedorn/develop/gatech/ml/mdp/taxi/experiments/data/Taxi-v3/rewards_{index}_stats_gamma0.7.csv")
            last_row = df.values[-1].tolist()
            steps.append(last_row[2])
            rewards.append(last_row[3])
        print("average reward",np.mean(rewards))
        print("average number of steps",np.mean(steps))

    def value_iteration():
        for gamma in gammas:
            df = pd.read_csv(f"data/Taxi-v3/value_iteration_value_fn_stats_gamma{gamma}.csv")
            df['gamma'] = gamma
            dfs.append(df)

        ds = 0
        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            error = data['span']
            gamma = data['gamma'][0]

            plt.plot(iterations, error)
            legends.append(f"gamma={gamma}")

        plt.title("taxi: value iteration error (span) vs iterations")
        plt.legend(legends, loc='upper right')
        plt.show()

        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            max_v = data[data.columns[3]]
            gamma = data[data.columns[8]][0]

            plt.plot(iterations, max_v, label="cumulative_reward")
            legends.append(f"gamma={gamma}")

        plt.title("taxi: value iteration max reward vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()

        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            time = data['Time']
            gamma = data[data.columns[8]][0]
            running_time = time.cumsum()

            plt.plot(iterations, running_time, label="cumulative_time")
            legends.append(f"gamma={gamma}")

        plt.title("forest: value iteration cumulative time vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()


    def q_learning():
        alphas = [0.4, 0.5, 0.6, 0.7]
        dfs = []
        for alpha in alphas:
            df = pd.read_csv(
                f"/Users/mwhagedorn/develop/gatech/ml/mdp/taxi/experiments/data/Taxi-v3/ql_training_gamma0.6_{alpha}_0.01.csv")
            dfs.append(df)

        ds = 0
        legends = []
        for ql in dfs:
            # every 10th row
            data = ql.iloc[:2000:100]
            iterations = data[data.columns[0]]
            error = data['penalty']
            alpha = data['alpha'][0]

            plt.plot(iterations, error)
            legends.append(f"alpha={alpha}")

        plt.title("Taxi: Q Learning penalties vs iterations")
        plt.legend(legends, loc='upper right')
        plt.show()
        dfs = []
        alphas = [0.4]
        for alpha in alphas:
            for episode in range(100):
                df = pd.read_csv(
                    f"/Users/mwhagedorn/develop/gatech/ml/mdp/taxi/experiments/data/Taxi-v3/rewards_0_stats_gamma{alpha}.csv")
                dfs.append(df)

        steps= []
        rewards = []
        for df in dfs:
            data = df.tail(1)
            step = data.values[0][0]
            reward = data.values[0][3]
            steps.append(step)
            rewards.append(reward)

        print("ave steps",np.mean(steps))
        print("ave rewards",np.mean(rewards))
    #policy_iteration()
    #value_iteration()
    q_learning()



create_graphs()