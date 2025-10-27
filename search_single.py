import traceback 
from multiprocessing import Pool, Manager
from vrp.data_utils import read_instance, mdvrp_to_plot_solution
from pyvrp import read as pyvrp_read
from pyvrp.plotting import plot_solution
from copy import deepcopy
import numpy as np
import torch
import repair
import time
import math
import search
import queue as pyqueue
#from plot.plot import plot_instance
import os
import csv
import logging

log = logging.getLogger(__name__)

def lns_single_seach_job(args):
    try:
        id, config, instance_path, model_path, queue_jobs, queue_results, pkl_instance_id = args
        rng = np.random.default_rng(id)
        operator_pairs = search.load_operator_pairs(model_path, config)
        instance = read_instance(instance_path, pkl_instance_id)

        T_min = config.lns_t_min

        # Repeat until the process is terminated
        while True:
            solution, incumbent_cost = queue_jobs.get() #block if necessary until an item is available
            incumbent_solution = deepcopy(solution)
            cur_cost = np.inf
            instance.solution = solution
            start_time_reheating = time.time()

            # Create a batch of copies of the same instances that can be repaired in parallel
            #note: lns_batch_size indicates multiple copies of the same vrp instance
            instance_copies = [deepcopy(instance) for _ in range(config.lns_batch_size)]

            iter = -1
            # Repeat until the time limit of one reheating iteration is reached
            while time.time() - start_time_reheating < config.lns_timelimit / config.lns_reheating_nb:
                iter += 1

                # Set the first config.lns_Z_param percent of the instances/solutions in the batch
                # to the last accepted solution
                for i in range(int(config.lns_Z_param * config.lns_batch_size)):
                    instance_copies[i] = deepcopy(instance)

                # Select an LNS operator pair (destroy + repair operator)
                selected_operator_pair_id = np.random.randint(0, len(operator_pairs))
                actor = operator_pairs[selected_operator_pair_id].model
                destroy_procedure = operator_pairs[selected_operator_pair_id].destroy_procedure
                p_destruction = operator_pairs[selected_operator_pair_id].p_destruction

                # Destroy instances
                search.destroy_instances(
                    rng                 = rng,
                    instances           = instance_copies, 
                    destroy_procedure   = destroy_procedure,
                    destruction_p       = p_destruction
                    )

                # Repair instances
                for i in range(int(len(instance_copies) / config.lns_batch_size)):
                    with torch.no_grad():
                        repair.repair(
                            instances   = instance_copies[i * config.lns_batch_size: (i + 1) * config.lns_batch_size],
                            actor       = actor, 
                            config      = config,
                            rng=rng,
                            )

                costs = [inst.get_costs_memory(config.round_distances) for inst in instance_copies]
                # Calculate the T_max and T_factor values for simulated annealing in the first iteration
                if iter == 0:
                    q75, q25 = np.percentile(costs, [75, 25])
                    T_max = q75 - q25
                    T_factor = -math.log(T_max / T_min)

                min_costs = min(costs)
                #print(f"DEBUG: job {id} -> min_costs: {min_costs}")

                # Update incumbent if a new best solution is found
                if min_costs <= incumbent_cost:
                    incumbent_solution = deepcopy(instance_copies[np.argmin(costs)].solution)
                    incumbent_cost = min_costs

                # Calculate simulated annealing temperature
                T = T_max * math.exp(
                    T_factor * (time.time() - start_time_reheating) / (config.lns_timelimit / config.lns_reheating_nb))

                # Accept a solution if the acceptance criteria is fulfilled
                if min_costs <= cur_cost or np.random.rand() < math.exp(-(min(costs) - cur_cost) / T):
                    instance.solution = instance_copies[np.argmin(costs)].solution
                    cur_cost = min_costs

            queue_results.put([incumbent_solution, incumbent_cost])

    except Exception as e:
        print("Exception in lns_single_search job: {0}".format(e))
        traceback.print_exc()   

def has_vehicles_line(instance_path: str) -> bool:
    """
    Determines wether the instance_path has line 'VEHICLES : INF' or not
    """
    lines = []
    with open(instance_path, 'r') as f:
        lines = f.readlines() 
    for line in lines:
        if line == 'VEHICLES : INF\n':
            return True
    return False


def lns_single_search_mp(instance_path, timelimit, config, model_path, pkl_instance_id=None, plot_initial_sol=False, run_id=None):
    instance = read_instance(instance_path, pkl_instance_id)
    instance.create_initial_solution()
    if plot_initial_sol:
        sol = mdvrp_to_plot_solution(instance)
        # if file has 'VEHICLES : INF' line
        data, temp_filename = None, None
        if has_vehicles_line(instance_path):
            #  create temp file without line
            temp_filename = f"{instance_path}_temp"
            # load data from temp file
            with open(instance_path) as f:
                lines = f.readlines()
                lines = [l for l in lines if l != 'VEHICLES : INF\n']
                with open(temp_filename, "w+") as f1:
                    f1.writelines(lines)
            data = pyvrp_read(temp_filename)
        else: 
            data = pyvrp_read(instance_path) 
        #plot solution
        plot_solution(sol, data, plot_clients=True)
        # delete temp file
        if temp_filename is not None:
            os.remove(temp_filename)

    dir_path = os.path.dirname(model_path)
    #plot_instance(instance, f"{dir_path}/initial_snapshot.png")
    #incumbent_costs = instance.get_costs(config.round_distances)
    incumbent_costs = instance.get_costs(config.round_distances)
    instance.verify_solution(config)

    start_time = time.time()
    m = Manager()
    queue_jobs = m.Queue()
    queue_results = m.Queue()
    pool = Pool(processes=config.lns_nb_cpus)
    log.debug(f"Simulated Annealing reheating iteration duration: {config.lns_timelimit/config.lns_reheating_nb}")
    pool.map_async(lns_single_seach_job,
                   [(i, config, instance_path, model_path, queue_jobs, queue_results, pkl_instance_id) for i in
                    range(config.lns_nb_cpus)])

    objective_trace = []
    objective_trace.append((0.0, float(incumbent_costs)))

    # Distribute starting solution to search processes
    for i in range(config.lns_nb_cpus):
        queue_jobs.put([instance.solution, incumbent_costs])
    
    while time.time() - start_time < timelimit:
        # Receive the incumbent solution from a finished search process (reheating iteration finished)
        result = queue_results.get()
        if result != 0:
            cost = result[1]
            objective_trace.append((time.time() - start_time, float(cost)))
            if cost < incumbent_costs:
                incumbent_costs = cost
                instance.solution = result[0]
        # Distribute incumbent solution to search processes
        queue_jobs.put([instance.solution, incumbent_costs])
    pool.terminate()
    duration = time.time() - start_time
    instance.verify_solution(config)
    log.info("Final solution:")
    for el in instance.solution:
        log.info(el)
    log.info("\n")

    trace_path = os.path.join(dir_path, f"objective_trace_{os.path.basename(instance_path).replace(os.sep, '_')}.csv")
    print(f"Printed objective trace to: {trace_path}")
    log.info(f"Printed objective trace to: {trace_path}")
    with open(trace_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "incumbent_cost"])
        w.writerows(objective_trace)
    # save copy of objective trace to config.output_path
    copy_of_trace_path = os.path.join(config.output_path, f"objective_trace_{os.path.basename(instance_path).replace(os.sep, '_')}.csv")
    with open(copy_of_trace_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "incumbent_cost"])
        w.writerows(objective_trace)
    log.info(f"Printed objective trace to: {copy_of_trace_path}")

    # plot final instance 
    #plot_path = f"{dir_path}/final_snapshot.png"
    ##plot_instance(instance, plot_path)
    #print(f"Plotted result to {plot_path}")
    #return instance.get_costs(config.round_distances), duration, instance
    return instance.get_costs(config.round_distances), duration, instance
