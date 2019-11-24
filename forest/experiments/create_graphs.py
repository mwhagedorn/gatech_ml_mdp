import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# If Gamma is closer to one, the agent will consider future rewards with greater weight, ...
def create_graphs():
    gammas = [0.3, 0.6, 0.7, 0.9]
    dfs = []
    # style
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')

    def policy_iteration():
        for gamma in gammas:
            df = pd.read_csv(f"data/forest_policy_iteration_{gamma}_run_stats.csv")
            df['gamma'] = gamma
            dfs.append(df)

        ds = 0
        legends= []
        for data in dfs:
            iterations = data[data.columns[0]]
            error = data[data.columns[4]]
            gamma = data[data.columns[8]][0]

            plt.plot(iterations,error)
            legends.append(f"gamma={gamma}")

        plt.title("forest: policy iteration error vs iterations")
        plt.legend(legends, loc='upper right')
        plt.show()


        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            max_v = data[data.columns[3]]
            gamma = data[data.columns[8]][0]

            plt.plot(iterations, max_v, label="cumulative_reward")
            legends.append(f"gamma={gamma}")

        plt.title("forest: policy iteration max reward vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()

        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            time = data['Time']
            gamma = data[data.columns[8]][0]
            running_time = time.cumsum()

            plt.plot(iterations,running_time,  label="cumulative_time")
            legends.append(f"gamma={gamma}")

        plt.title("forest: policy iteration cumulative time vs iterations")
        plt.legend(legends, loc='lower right')
        plt.show()

    def value_iteration():
        for gamma in gammas:
            df = pd.read_csv(f"data/forest_value_iteration_{gamma}_run_stats.csv")
            df['gamma'] = gamma
            dfs.append(df)

        ds = 0
        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            error = data[data.columns[4]]
            gamma = data[data.columns[8]][0]

            plt.plot(iterations, error)
            legends.append(f"gamma={gamma}")

        plt.title("forest: value iteration error vs iterations")
        plt.legend(legends, loc='upper right')
        plt.show()

        legends = []
        for data in dfs:
            iterations = data[data.columns[0]]
            max_v = data[data.columns[3]]
            gamma = data[data.columns[8]][0]

            plt.plot(iterations, max_v, label="cumulative_reward")
            legends.append(f"gamma={gamma}")

        plt.title("forest: value iteration max reward vs iterations")
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
        df = pd.read_csv(f"data/forest_ql_rolling_errs_0.9.csv")
        error = df[df.columns[1]]

        plt.plot(error)
        plt.title("forest: q_learning average error vs iterations")
        plt.show()

        pass



    #policy_iteration()
    #value_iteration()
    q_learning()

create_graphs()