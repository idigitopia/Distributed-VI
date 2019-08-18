import argparse
import pickle as pk
import time

import utils as utils
from env_frozen_lake import FrozenLakeEnvDynamic, save_and_print_results, get_performance_log, get_tran_reward_dict, process_log_data
from vi_engine_s import distributed_value_iteration, simple_value_iteration

ENV_MAP = {"frozen_lake": FrozenLakeEnvDynamic}
VI_ENGINE_MAP = {'distributed': distributed_value_iteration, 'simple': simple_value_iteration}


def run_experiment(args):
    # House Keeping
    results_dir = "results/" + str(args.env_name) +"_" +str(args.experiment_id)+"/"
    utils.create_hierarchy(results_dir)

    # Define Environment
    Env = ENV_MAP[args.env_name]
    full_map_size = (args.map_size, args.map_size)
    try:
        env = Env(full_map_size) if not args.load_env else pk.load(
            open(results_dir + "env" + str(args.map_size) + ".pk", "rb"))
    except:
        env = Env(full_map_size)

    pk.dump(env, open(results_dir + "env" + str(args.map_size) + ".pk", "wb"))

    # Get transition and reward dictionaries
    st = time.time()

    list_of_states = list(range(env.GetStateSpace()))
    list_of_actions = list(range(env.GetActionSpace()))
    tran_dict, reward_dict = get_tran_reward_dict(env)

    matrix_time = time.time() - st
    print("time taken to get transition matrix:", matrix_time)

    # Solve using VI
    st = time.time()
    workers_num = args.workers
    vi_engine_func = VI_ENGINE_MAP[args.vi_engine]

    v, pi = vi_engine_func(S=list_of_states,
                           A=list_of_actions,
                           reward_dict=reward_dict,
                           tran_dict=tran_dict,
                           beta=0.99,
                           epsilon=0.01,
                           workers_num=workers_num,
                           verbose=args.verbose)
    v, pi = list(v.values()), list(pi.values())

    vi_time = time.time() - st
    print("time taken:", vi_time)

    # Process Results and Add to plot
    save_and_print_results(v, pi, MAP=env.map_grid, env=env, beta=0.99, name=args.vi_engine, results_dir=results_dir)
    performance_log = get_performance_log(args.vi_engine, full_map_size, workers_num, vi_time, matrix_time, results_dir,
                                      store=True)
    utils.createPlotlyPlotFor(process_log_data(performance_log),
                              x_label=" Number of States",
                              y_label="Runtime (Seconds)",
                              title="Distributed VI",
                              show=False,
                              save=True,
                              fileName=results_dir + "/Distributed VI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--env_name", help="Choice of Environment to solve",
                        choices=['frozen_lake'], default='frozen_lake')
    parser.add_argument('-exp_id', "--experiment_id", help="Experiment id for new result folder", default='R1')
    parser.add_argument('-vi', "--vi_engine", help="Choice of VI engine to use",
                        choices=['distributed', 'simple'], default='distributed')
    parser.add_argument("-w", "--workers", help="Number of Workers", type=int, default=4)
    parser.add_argument("-m", "--map_size", help="map size(s), comma separated", default='100')
    parser.add_argument("--load_env", help="load environment from cache ?", action="store_true", default=True)
    parser.add_argument("--verbose", help="print errors", action="store_true", default=False)
    parser.add_argument("-r", "--num_of_runs", help="Number of full pipeline runs", type=int, default=10)
    args = parser.parse_args()

    map_sizes = [int(i) for i in args.map_size.split(',')]  # [10, 32, 72, 100, 225, 320, 500, 708, 868, 1000]

    for map_size in map_sizes:
        for i in range(args.num_of_runs):
            args.map_size = map_size
            run_experiment(args)

