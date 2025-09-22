# NOTE: if this code fails for unhashable numpy type, add following line in vrplib parse_section function:

#        if name == "vehicles_depot":
#            data = np.array([row[0] for row in rows])
#        else:
#            data = np.array([row[1:] for row in rows])


import argparse
from pyvrp import Model, read, Result
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution, plot_coordinates
import matplotlib.pyplot as plt
import numpy as np
from vrplib.read import read_instance
from pathlib import Path
from vrp.mdvrp_problem import MDVRPInstance
import os
from typing import Union
from tqdm import tqdm

def eval_single(args):
    data = read_instance(args.instance_path)
    result, m = run_pyvrp_on_instance(
                inst        = data, 
                only_cost   = False,
                display     = True
                )
    print(result)
    
    if args.plot_solution:
        plot_solution(result.best, m.data())
    
    _, ax = plt.subplots(figsize=(8, 8))
    plot_solution(result.best, m.data(), ax=ax)
    plt.show()


def run_pyvrp_on_instance(inst: MDVRPInstance, only_cost: bool = False, display: bool = False) -> Union[float, tuple]:
    """
    Runs PyVRP on instance. Returns only cost is only_cost is True, otherwise returns Result.
    """
    m = Model()
    num_depots = len(inst['depot'])
    depots = []
    for d in inst['depot']:
        depot = m.add_depot(x=inst['node_coord'][d][0], y=inst['node_coord'][d][1])
        depots.append(depot)
    
    if str(inst['vehicles']) != 'inf':
        print(f"DEBUG: read inst['vehicles'] : {inst['vehicles']}, type: {type(inst['vehicles'])}")
        keys, counts = np.unique(inst['vehicles_depot'], return_counts=True)
        keys = keys - 1
        keys = keys.tolist()
        counts = counts.tolist()
        depot_num_vehicles =  dict(zip(keys, counts))
    else:
        depot_num_vehicles = {}
        for i, d in enumerate(inst['depot']):
            depot_num_vehicles[i] = inst['dimension'] - num_depots
    
    for i, d in enumerate(inst['depot']):
        m.add_vehicle_type(
            num_available   = depot_num_vehicles[int(d)],
            capacity        = inst['capacity'],
            start_depot     = depots[i],
            end_depot       = depots[i],
        )
    
    clients = [
        m.add_client(
            x=int(inst['node_coord'][idx][0]),
            y=int(inst['node_coord'][idx][1]),
            delivery=int(inst['demand'][idx]),
        )
        for idx in range(num_depots, len(inst['node_coord']))
    ]
    
    locations = depots + clients
    
    for frm_idx, frm in enumerate(locations):
        for to_idx, to in enumerate(locations):
            #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
            distance = np.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2) 
            m.add_edge(frm, to, distance=distance)
    
    result = m.solve(stop=MaxRuntime(args.max_time), display=display)
    if only_cost:    
        return result.best.distance_cost() 
    else:
        return result, m

def read_instances(dir_path: Path) -> list:
    list_of_files = os.listdir(dir_path)
    cwd = os.getcwd()
    instances = []
    for inst in list_of_files:
        data_path = os.path.join(cwd, dir_path, inst)
        if os.path.splitext(data_path)[1] != '.mdvrp':
            continue
        data = read_instance(data_path)
        instances.append(data)

    return instances


def eval_batch(args):
    instances = read_instances(args.dir_path)
    results = [] 
    for i, inst in tqdm(enumerate(instances)):
        cost = run_pyvrp_on_instance(
                inst        = inst,
                only_cost   = True,
                display     = False)
        results.append([i, cost])

    output_filename = os.path.join(args.output_dir, "pyrvp_eval_batch.txt" ) 
    with open(output_filename, 'a') as f:
        for i, el in enumerate(results):
            f.write(f"{i},{el}\n")

    print(f"Written cost results of PyVRP in file: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyVRP model execution")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['eval_single', 'eval_batch'],
        required=True,
        help="Mode of operation. Valid choices are 'eval_single' and 'eval_batch'"
    )
    parser.add_argument( 
        "--instance_path",
        default=None,
        type=str,
        help="Path to one VRPLIB file",
    )
    parser.add_argument( 
        "--dir_path",
        default=None,
        type=str,
        help="Path to dir with VRPLIB files",
    )
    parser.add_argument( 
        "--output_dir",
        default=None,
        type=str,
        help="Path to dir where results should be solved",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=10.0,
        help="Max runtime per instance in seconds (float).",
    )
    parser.add_argument(
        "--plot_solution", "--plot-solution",
        dest='plot_solution',
        action='store_true',
        help="Plot instance solution.",
    )
    args = parser.parse_args()
    
    if args.mode == 'eval_batch':
        assert args.dir_path is not None, f"Missing argument dir_path in 'eval_batch' mode"
        assert args.output_dir is not None, f"Missing argument output_dir in 'eval_batch' mode"

        eval_batch(args)
    
    elif args.mode == 'eval_single':
        assert args.instance_path is not None, f"Missing argument instance_path in 'eval_single' mode"
        eval_single(args)
