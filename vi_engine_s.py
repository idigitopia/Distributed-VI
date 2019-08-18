import numpy as np
import ray

ray.shutdown()
ray.init()


# A : Action Space
# S : State Space 

@ray.remote
class VI_worker_class(object):
    def __init__(self, list_of_actions, tran_dict, reward_dict, beta, backup_states, true_action_prob=0.8,
                 unknown_value=0):
        self.backup_states = backup_states
        self.list_of_actions = list_of_actions
        self.tran_dict = tran_dict
        self.reward_dict = reward_dict
        self.beta = beta
        self.unknown_value = unknown_value  # Default Value for any states that do not have transitions defined.

        self.true_action_prob = true_action_prob
        self.slip_prob = 1 - self.true_action_prob
        self.slip_action_prob = self.slip_prob / len(self.list_of_actions)

    def compute(self, V_t, backup_states=None):
        """
        
        :param V_t: Value Vector at t
        :return: 
        """
        backup_states = backup_states or self.backup_states

        V_tplus1 = {s: 0 for s in backup_states}
        max_vals = {s: float("-inf") for s in backup_states}

        max_error = 0

        for s in backup_states:
            for a in self.tran_dict[s]:
                expected_ns_val = 0
                for ns in self.tran_dict[s][a]:
                    try:
                        expected_ns_val += self.tran_dict[s][a][ns] * V_t[ns]
                    except:
                        expected_ns_val += self.tran_dict[s][a][ns] * self.unknown_value

                expect_s_val = self.reward_dict[s][a] + self.beta * expected_ns_val
                max_vals[s] = max(max_vals[s], expect_s_val)
                V_tplus1[s] += self.slip_action_prob * expect_s_val
            V_tplus1[s] += (self.true_action_prob - self.slip_action_prob) * max_vals[s]

            max_error = max(max_error, abs(V_tplus1[s] - V_t[s]))

        return V_tplus1, max_error


def distributed_value_iteration(S, A, reward_dict, tran_dict, seed_value=None, unknown_value=0, true_action_prob=0.8,
                                beta=0.99, epsilon=0.01, workers_num=4, verbose=True):
    # Split the state space evenly to be distributed to VI workers
    state_chunks = [a.tolist() for a in np.array_split(np.array(S), workers_num)]
    V_t = {s: 0 for s in S} if seed_value is None else seed_value

    # Make VI workers
    workers_list = [VI_worker_class.remote(list_of_actions=A,
                                           tran_dict=tran_dict,
                                           reward_dict=reward_dict,
                                           beta=beta,
                                           backup_states=state_chunk,
                                           unknown_value=unknown_value,
                                           true_action_prob=true_action_prob)
                    for state_chunk in state_chunks]

    # Do VI computation
    error = float('inf')
    while error > epsilon:
        object_list = [workers_list[i].compute.remote(V_t) for i in range(workers_num)]
        error_list = []
        for i in range(workers_num):
            finish_id = ray.wait(object_list, num_returns=1, timeout=None)[0][0]
            object_list.remove(finish_id)
            V_tplus1, error = ray.get(finish_id)

            V_t.update(V_tplus1)
            error_list.append(error)

            if (verbose):
                print("Error:", error)

        error = max(error_list)

    pi = get_pi_from_value(V_t, A, tran_dict, reward_dict, beta)

    return V_t, pi


def simple_value_iteration(S, A, reward_dict, tran_dict, seed_value=None, unknown_value=0, true_action_prob=0.8,
                           beta=0.99, epsilon=0.01, workers_num=4, verbose=True):
    slip_prob = 1 - true_action_prob
    slip_action_prob = slip_prob / len(A)

    V_t = {s: 0 for s in S} if seed_value is None else seed_value
    error = float("inf")

    while error > epsilon:
        V_tplus1 = {s: 0 for s in S}
        max_vals = {s: float("-inf") for s in S}

        max_error = 0

        for s in S:
            for a in tran_dict[s]:
                expected_ns_val = 0
                for ns in tran_dict[s][a]:
                    try:
                        expected_ns_val += tran_dict[s][a][ns] * V_t[ns]
                    except:
                        expected_ns_val += tran_dict[s][a][ns] * unknown_value

                expect_s_val = reward_dict[s][a] + beta * expected_ns_val
                max_vals[s] = max(max_vals[s], expect_s_val)
                V_tplus1[s] += slip_action_prob * expect_s_val
            V_tplus1[s] += (true_action_prob - slip_action_prob) * max_vals[s]

            max_error = max(max_error, abs(V_tplus1[s] - V_t[s]))

        V_t.update(V_tplus1)
        error = max_error

        if (verbose):
            print("Error:", error)

    pi = get_pi_from_value(V_t, A, tran_dict, reward_dict, beta)

    return V_t, pi


def get_pi_from_value(V, list_of_actions, tran_dict, reward_dict, beta):
    v_max = {s: float('-inf') for s in V}
    pi = {}

    for s in V:
        for a in tran_dict[s]:
            expected_val = 0
            for ns in tran_dict[s][a]:
                try:
                    expected_val += tran_dict[s][a][ns] * V[ns]
                except:
                    expected_val += tran_dict[s][a][ns] * 0
            expect_s_val = reward_dict[s][a] + beta * expected_val
            if expect_s_val > v_max[s]:
                v_max[s] = expect_s_val
                pi[s] = a

    return pi
