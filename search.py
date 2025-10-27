import logging
import numpy as np
import os
import time
import torch
import search_single
import tempfile 
from vrp.data_utils import read_instances_pkl, read_instance, mdvrp_to_plot_solution
import glob
import search_batch
from actor import VrpActorModel
from dummy_model import dummy_model
from vrp.mdvrp_problem import MDVRPInstance
from tqdm import trange
from pyvrp.plotting import plot_solution
from pyvrp import read as pyvrp_read
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LnsOperatorPair:
    def __init__(self, model, destroy_procedure, p_destruction):
        self.model = model
        self.destroy_procedure = destroy_procedure
        self.p_destruction = p_destruction
    def __str__(self):
        return f"Destroy_procedure: {self.destroy_procedure} | P_destruction: {self.p_destruction}"


def destroy_instances(rng, instances, destroy_procedure=None, destruction_p=None):
    for instance in instances:
        if destroy_procedure == "R":
            instance.destroy_random(destruction_p, rng=rng)
        elif destroy_procedure == "P":
            instance.destroy_point_based(destruction_p, rng=rng)
        elif destroy_procedure == "T":
            instance.destroy_tour_based(destruction_p, rng=rng)


def load_operator_pairs(path, config):
    if path.endswith('.pt'):
        model_paths = [path]
    else:
        model_paths = glob.glob(os.path.join(path, '*.pt'))

    if not model_paths:
        raise Exception(f"No operators found in {path}")

    lns_operator_pairs = []
    for model_path in model_paths:
        model_data = torch.load(model_path, config.device)

        actor = VrpActorModel(config.device, hidden_size=config.pointer_hidden_size).to(
            config.device)
        actor.load_state_dict(model_data['parameters'])
        actor.eval()

        operator_pair = LnsOperatorPair(actor, model_data['destroy_operation'], model_data['p_destruction'])
        lns_operator_pairs.append(operator_pair)
    return lns_operator_pairs


def evaluate_batch_search(config, model_path):
    assert model_path is not None, 'No model path given'

    logger.info('### Batch Search ###')
    logger.info('Starting search')
    start_time = time.time()

    results = search_batch.lns_batch_search_mp(config, model_path)
    runtime = (time.time() - start_time)
    instance_id, costs, iterations = [], [], []
    for r in results:
        instance_id.extend(list(range(len(r[1]) * r[0], len(r[1]) * (r[0] + 1))))
        costs.extend(r[1])
        iterations.append(r[2])

    path = os.path.join(config.output_path, "search", "nlns_batch_search_results.txt")
    np.savetxt(path, np.column_stack((instance_id, costs)), delimiter=',', fmt=['%i', '%f'])
    print(f"Saved results of batch search in {path}")
    logger.info(
        f"Test set costs: {np.mean(costs):.3f} Total Runtime (s): {runtime:.1f} Iterations: {np.mean(iterations):.1f}")

def evaluate_single_search(config, model_path, instance_path, run_id=None):
    assert model_path is not None, 'No model path given'
    assert instance_path is not None, 'No instance path given'

    # if instance_path has VEHICLES: INF -> Remove this line
    if os.path.isfile(instance_path):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            with open(instance_path, "r") as infile:
                for line in infile:
                    if line.strip() != "VEHICLES : INF":
                        tmp.write(line)
        os.replace(tmp.name, instance_path)

    instance_names, instance_ids, costs, durations = [], [], [], []
    logger.info("### Single instance search ###")

    if instance_path.endswith(".vrp") or instance_path.endswith(".sd") or instance_path.endswith(".mdvrp"):
        logger.info("Starting solving a single instance")
        instance_files_path = [instance_path]
    elif instance_path.endswith(".pkl"):
        instance_files_path = [instance_path] * len(read_instances_pkl(instance_path))
        logger.info("Starting solving a .pkl instance set")
    elif os.path.isdir(instance_path):
        instance_files_path = [os.path.join(instance_path, f) for f in os.listdir(instance_path)]
        logger.info("Starting solving all instances in directory")
    else:
        raise Exception("Unknown instance file format.")

    for i, instance_path in enumerate(tqdm(instance_files_path)):
        if instance_path.endswith(".pkl") or instance_path.endswith(".vrp") or instance_path.endswith(".sd") or instance_path.endswith(".mdvrp"):
            for jj in range(config.nb_runs):
                cost, duration, final_instance = search_single.lns_single_search_mp(instance_path, config.lns_timelimit, config, model_path, i, run_id=run_id, plot_initial_sol=False)
                instance_names.append(instance_path)
                instance_ids.append(i)
                costs.append(cost)
                durations.append(duration)

    output_path_with_times = os.path.join(config.output_path, "search", 'nlns_batch_search_results_with_times.txt')
    output_path = os.path.join(config.output_path, "search", 'nlns_batch_search_results.txt')
    results_with_times= np.array(list(zip(instance_names, costs, durations)))
    results = np.array(list(zip(instance_ids, costs)))

    np.savetxt(output_path_with_times, results_with_times, delimiter=',', fmt=['%s', '%s', '%s'], header="name, cost, runtime")
    np.savetxt(output_path, results, delimiter=',',)

    logger.info(
        f"NLNS single search evaluation results: Total Nb. Runs: {len(costs)}, "
        f"Mean Costs: {np.mean(costs):.3f} Mean Runtime (s): {np.mean(durations):.1f}")

    if config.plot_solution:
        sol = mdvrp_to_plot_solution(final_instance)
        data = pyvrp_read(instance_path) 
        plot_solution(sol, data, plot_clients=True)

    #print on terminal
    tours = final_instance.solution
    tours = [t for t in tours if len(t) > 1]
    logger.info(f"Solution results\n==============")
    logger.info(f"\t# routes: {len(tours)}")
    logger.info(f"\t# clients: {final_instance.nb_customers}")
    logger.info(f"\t# n_depots: {final_instance.n_depots}")
    logger.info(f"\t# depot indices: {final_instance.depot_indices}")
    #logger.info(f"\t# costs: {final_instance.get_costs(True)}")
    logger.info(f"\t# costs: {final_instance.get_costs()}")

def evaluate_multi_depot_search(config, instance_path):
    assert instance_path is not None, "No instance path given"
    assert instance_path.endswith(".mdvrp"), "Wrong instance type (doesn't end with '.mdvrp')"

    instance_names, costs, durations = [], [], []
    logger.info("### Single instance search for multi depot case###")

    if instance_path.endswith(".mdvrp"):
        logger.info("Starting solving a single instance")
        instance_files_path = [instance_path]
    #elif instance_path.endswith(".pkl"):
    #    instance_files_path = [instance_path] * len(read_instances_pkl(instance_path))
    #    logger.info("Starting solving a .pkl instance set")
    elif os.path.isdir(instance_path):
        instance_files_path = [os.path.join(instance_path, f) for f in os.listdir(instance_path)]
        logger.info("Starting solving all instances in directory")
    else:
        raise Exception("Unknown instance file format.")

    for i, instance_path in enumerate(instance_files_path):
        if instance_path.endswith(".mdvrp"):
            for _ in range(config.nb_runs):
                #cost, duration = search_single.lns_single_search_mp(instance_path, config.lns_timelimit, config,
                #                                                    model_path, i)

                cost, duration = dummy_model(
                        instance_path   = instance_path,
                        config          = config, 
                        plot_image     = True)
                instance_names.append(instance_path)
                costs.append(cost)
                durations.append(duration)

    output_path = os.path.join(config.output_path, "search", 'results.txt')
    results = np.array(list(zip(instance_names, costs, durations)))
    np.savetxt(output_path, results, delimiter=',', fmt=['%s', '%s', '%s'], header="name, cost, runtime")

    logger.info(
        f"NLNS single search evaluation results: Total Nb. Runs: {len(costs)}, "
        f"Mean Costs: {np.mean(costs):.3f} Mean Runtime (s): {np.mean(durations):.1f}")
