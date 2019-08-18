import pickle
import sys
from statistics import mean
import matplotlib.pyplot as plt
import pickle as pk
import math
import numpy as np
import time

class FrozenLakeEnvDynamic():

    def __init__(self, map_size: tuple):
        self.map_grid = generate_map(map_size)
        self.map_size = map_size
        self.nrow = map_size[0]
        self.ncol = map_size[1]
        self.terminal_row = self.nrow
        self.terminal_col = 0
        self.state = 0

    def get_state(self, row, col):
        return row * self.ncol + col

    def get_row_col(self, state):
        return (math.floor(state / self.ncol), int(state % self.nrow))

    def get_next_row_col(self, row, col, a):
        if (row == self.terminal_row and col == self.terminal_col):
            return (row, col)

        if (self.map_grid[row][col] == "G" or self.map_grid[row][col] == "H"):
            return (self.terminal_row, self.terminal_col)

        if a == 3:
            col = max(col - 1, 0)
        elif a == 2:
            row = min(row + 1, self.nrow - 1)
        elif a == 1:
            col = min(col + 1, self.ncol - 1)
        elif a == 0:
            row = max(row - 1, 0)
        return (row, col)

    def get_all_states(self):
        all_states = list(range(self.map_size[0] * self.map_size[1] + 1))
        return all_states

    def GetStateSpace(self):
        return self.map_size[0] * self.map_size[1] + 1

    def get_all_actions(self):
        """
            0:UP
            1:Right
            2:Down
            3:Left
        """
        return list(range(4))

    def GetActionSpace(self):
        """
            0:UP
            1:Right
            2:Down
            3:Left
        """
        return 4

    def GetSuccessors(self, s, a):
        successors = []
        row, col = self.get_row_col(s)

        for r_a in self.get_all_actions():
            next_row, next_col = self.get_next_row_col(row, col, r_a)
            next_s = self.get_state(next_row, next_col)
            successors.append([next_s, 0.7] if r_a == a else [next_s, 0.1])
        return successors

    def GetTransitReward(self, s, a):
        row, col = self.get_row_col(s)

        if (row == self.terminal_row and col == self.terminal_col):
            reward = 0
        elif (self.map_grid[row][col] == "H"):
            reward = -1000
        elif (self.map_grid[row][col] == "G"):
            reward = 1000
        elif (self.map_grid[row][col] == "F"):
            reward = -1
        else:
            reward = -1

        return reward

    def reset(self):
        self.state = 0

    def step(self, a):
        done = False

        row, col = self.get_row_col(self.state)
        next_row, next_col = self.get_next_row_col(row, col, a)
        next_state = self.get_state(next_row, next_col)

        if (next_row == self.terminal_row and next_col == self.terminal_col):
            done = True
        elif self.map_grid[next_row][next_col] == "G" or self.map_grid[next_row][next_col] == "H":
            done = True

        reward = self.GetTransitReward(self.state, a)
        info = {}
        self.state = next_state
        return next_state, reward, done, info




# Transition Dictionary helper functions
def get_tran_reward_dict(env):
    list_of_states = list(range(env.GetStateSpace()))
    list_of_actions = list(range(env.GetActionSpace()))
    reward_dict = {}
    tran_dict = {}

    # get transition dictionary

    for s in list_of_states:
        reward_dict[s] = {}
        tran_dict[s] = {}
        for a in list_of_actions:
            reward_dict[s][a] = env.GetTransitReward(s, a)
            tran_dict[s][a] = {}
            successors = env.GetSuccessors(s, a)
            for ns, p in successors:
                tran_dict[s][a][ns] = p

    return tran_dict, reward_dict


# Map Helper Functions

def generate_row(length, h_prob):
    row = np.random.choice(2, length, p=[1.0 - h_prob, h_prob])
    row = ''.join(list(map(lambda z: 'F' if z == 0 else 'H', row)))
    return row


def generate_map(shape):
    """

    :param shape: Width x Height
    :return: List of text based map
    """
    h_prob = 0.1
    grid_map = []

    for h in range(shape[1]):

        if h == 0:
            row = 'SF'
            row += generate_row(shape[0] - 2, h_prob)
        elif h == 1:
            row = 'FF'
            row += generate_row(shape[0] - 2, h_prob)

        elif h == shape[1] - 1:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FG'
        elif h == shape[1] - 2:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FF'
        else:
            row = generate_row(shape[0], h_prob)

        grid_map.append(row)
        del row

    return grid_map




MAPS = {

    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "16x16": [
        "SFFFFFFFFHFFFFHF",
        "FFFFFFFFFFFFFHFF",
        "FFFHFFFFHFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFHHFFFFFFFHFFFH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFHFFF",
        "FFFFFHFFFFFFFFFH",
        "FFFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFHFFFFHF",
        "FFFFFFFFFFHFFFFF",
        "FFFHFFFFFFFFFFFG",
    ],

    "32x32": [
        'SFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFH',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFH',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFH',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFH',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFG',
    ],

    "64x64": [
        'SFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFFSFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFFFFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFHFFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHFFFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFH',
        'FFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFFSFFHFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFFFFHFHHFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFHFFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHFFFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFHFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFFFFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFG',
    ]
}



# Print and Evaluate Helper Functions

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=2)

def evaluate_policy(env, policy, trials=10):
    total_reward = 0
    #     epoch = 10
    max_steps = 500

    for _ in range(trials):
        steps = 0
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while (not done) and (steps < max_steps):
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward
            steps += 1
            # if(steps%100) == 0 :
            #     print(steps)
    return total_reward / trials


def evaluate_policy_discounted(env, policy, discount_factor, trials=10):
    epoch = 10
    reward_list = []
    max_steps = 500

    for _ in range(trials):
        steps = 0
        total_reward = 0
        trial_count = 0
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while (not done) and (steps < max_steps):
            observation, reward, done, info = env.step(policy[observation])
            total_reward += (discount_factor ** trial_count) * reward
            trial_count += 1
            steps += 1
            # if(steps%100) == 0 :
                # print(steps)
        reward_list.append(total_reward)

    return mean(reward_list)


def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size, map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size, map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor=beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size, map_size)))

    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))


def save_and_print_results(v, pi, MAP, env, beta, name, show_val=False, show_pi=False,
                           results_dir="results/frozen_lake/"):
    map_size = len(MAP)
    v_np, pi_np = np.array(v), np.array(pi)
    if (show_val):
        print("\nState Value:\n")
        print(np.array(v_np[:-1]).reshape((map_size, map_size)))
    if (show_pi):
        print("\nPolicy:\n")
        print(np.array(pi_np[:-1]).reshape((map_size, map_size)))

    avg_reward = evaluate_policy(env, pi)
    avg_discounted_reward = evaluate_policy_discounted(env, pi, discount_factor=beta)
    print("\nAverage reward: {}\n".format(avg_reward))
    print("Avereage discounted reward: {}\n".format(avg_discounted_reward))
    print("State Value image view:\n")

    plt.imsave(results_dir + "value_" + str(map_size) + ".png", rescale_data(np.array(v_np[:-1]).reshape((map_size, map_size))))
    pickle.dump(v, open(results_dir + name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(results_dir + name + "_" + str(map_size) + "_pi.pkl", "wb"))

    plot_and_save_policy_image(v,pi,MAP,results_dir)

    return avg_reward, avg_discounted_reward

def save_results(v, map_size):
    v_np = np.array(v)
    plt.imsave("latest_fig.png", np.array(v_np[:-1]).reshape((map_size, map_size)), dpi=400)

def rescale_data(data):
    scale = int(1000/len(data))
    new_data = np.zeros(np.array(data.shape) * scale)
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            new_data[j * scale: (j+1) * scale, k * scale: (k+1) * scale] = data[j, k]
    return new_data


def plot_and_save_policy_image(value, pi, MAP, results_dir="results/frozen_lake/"):
    best_value = np.array(value[:-1]).reshape(len(MAP), len(MAP))
    best_policy = np.array(pi[:-1]).reshape(len(MAP), len(MAP))

    print("\n\nBest Q-value and Policy:\n")
    fig, ax = plt.subplots()
    im = ax.imshow(best_value)

    for i in range(best_value.shape[0]):
        for j in range(best_value.shape[1]):
            if MAP[i][j] in 'GH':
                arrow = MAP[i][j]
            elif best_policy[i, j] == 0:
                arrow = '^'
            elif best_policy[i, j] == 1:
                arrow = '>'
            elif best_policy[i, j] == 2:
                arrow = 'V'
            elif best_policy[i, j] == 3:
                arrow = '<'
            if MAP[i][j] in 'S':
                arrow = 'S ' + arrow
            text = ax.text(j, i, arrow,
                           ha="center", va="center", color="black")

    cbar = ax.figure.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.savefig(results_dir + "policy_" + str(len(MAP)) + ".png",)  # , rescale_data(np.array(v_np[:-1]).reshape((map_size, map_size))))
    # plt.show()


def get_performance_log(v, m, w, VI_time, matrix_time, results_dir="results/frozen_lake/", store = True):
    """
    :param v: VI Engine
    :param m: Map Size
    :param w: Workers Number
    :param VI_time: Time taken for VI to complete
    :param matrix_time: Time taken to calculate transition and reward Dictionaries
    :param results_dir:
    :param store: Save to disk Flag
    :return:
    """
    try:
        performance_log = pk.load(open(results_dir + "performance_log.pk", "rb"))
    except:
        performance_log = {}
    # print("loading",performance_log)
    performance_log[v] = {} if v not in performance_log else performance_log[v]
    performance_log[v][w] = {} if w not in performance_log[v] else performance_log[v][w]
    performance_log[v][w][m] = {} if m not in performance_log[v][w] else performance_log[v][w][m]
    performance_log[v][w][m]["matrix_time"] = [] if "matrix_time" not in performance_log[v][w][m] else performance_log[v][w][m]["matrix_time"]
    performance_log[v][w][m]["VI_time"] = [] if "VI_time" not in performance_log[v][w][m] else performance_log[v][w][m]["VI_time"]
    performance_log[v][w][m]["matrix_time"].append(matrix_time)
    performance_log[v][w][m]["VI_time"].append(VI_time)

    if(store):
        pk.dump(performance_log, open(results_dir + "performance_log.pk", "wb"))

    return performance_log

def process_log_data(perf_log):
    data = []
    for vi_engine in perf_log:
        for worker_num  in perf_log[vi_engine]:
            data_x = []
            data_y = []
            for map_size in perf_log[vi_engine][worker_num]:
                num_of_states = map_size[0]**2
                avg_runtime = mean(perf_log[vi_engine][worker_num][map_size]["VI_time"])
                data_x.append(num_of_states)
                data_y.append(avg_runtime)
            data.append((data_x, data_y, vi_engine + str(worker_num)))
    return data