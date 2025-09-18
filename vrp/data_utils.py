import os
from copy import deepcopy
import numpy as np
from vrp.vrp_problem import VRPInstance
from vrp.mdvrp_problem import MDVRPInstance
import pickle
from tqdm import trange 
from typing import List, Union
from pyvrp import Solution, ProblemData, Route
from pyvrp import read as pyvrp_read
from typing import Iterable, List

class InstanceBlueprint:
    """Describes the properties of a certain instance type (e.g. number of customers)."""

    def __init__(self, nb_customers, depot_position, customer_position, nb_customer_cluster, demand_type, demand_min,
                 demand_max, capacity, grid_size, n_depots=1, vehicle_policy='RR'):
        self.nb_customers = nb_customers
        self.n_depots = n_depots 
        self.depot_position = depot_position
        self.customer_position = customer_position
        self.nb_customers_cluster = nb_customer_cluster
        self.demand_type = demand_type
        self.demand_min = demand_min
        self.demand_max = demand_max
        self.capacity = capacity
        self.grid_size = grid_size
        self.vehicle_policy = vehicle_policy

def get_blueprint(blueprint_name):
    instance_type = blueprint_name.split('_')[0]
    instance = blueprint_name.split('_')[1]
    if instance_type == "ALTR":
        import vrp.dataset_blueprints.ALTR
        return vrp.dataset_blueprints.ALTR.dataset[instance]
    elif instance_type == "XE":
        import vrp.dataset_blueprints.XE
        return vrp.dataset_blueprints.XE.dataset[instance]
    elif instance_type == "S":
        import vrp.dataset_blueprints.S
        return vrp.dataset_blueprints.S.dataset[instance]
    elif instance_type == "MD":
        import vrp.dataset_blueprints.MD
        return vrp.dataset_blueprints.MD.dataset[instance]
    raise Exception('Unknown blueprint instance')


def create_dataset(size, config, seed=None, create_solution=False, use_cost_memory=True):
    instances = []
    blueprints = get_blueprint(config.instance_blueprint)


    if config.problem_type == 'mdvrp':
        assert config.instance_blueprint in ['MD_1', 'MD_2', 'MD_3', 'MD_4', 'MD_5', 'MD_6', 'MD_7']

    #if seed is not None:
    #    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    for i in trange(size):
    #for i in range(size):
        if isinstance(blueprints, list):
            blueprint_rnd_idx = rng.integers(0, len(blueprints), 1).item()
            vrp_instance = generate_Instance(blueprints[blueprint_rnd_idx], use_cost_memory, rng)
        else:
            vrp_instance = generate_Instance(blueprints, use_cost_memory, rng)
        instances.append(vrp_instance)
        if create_solution:
            vrp_instance.create_initial_solution()
    return instances


def generate_Instance(blueprint, use_cost_memory, rng):
    if blueprint.n_depots == 1:
        depot_position = get_depot_position(blueprint, rng)
        customer_position = get_customer_position(blueprint, rng)
        demand = get_customer_demand(blueprint, customer_position, rng)
        original_locations = np.insert(customer_position, 0, depot_position, axis=0)
        demand = np.insert(demand, 0, 0, axis=0)

        if blueprint.grid_size == 1000:
            locations = original_locations / 1000
        elif blueprint.grid_size == 1000000:
            locations = original_locations / 1000000
        else:
            assert blueprint.grid_size == 1
            locations = original_locations

        vrp_instance = VRPInstance(blueprint.nb_customers, locations, original_locations, demand, blueprint.capacity,
                                   use_cost_memory)
        return vrp_instance

    elif blueprint.n_depots > 1: #mdvrp
        depot_positions = []
        for d in range(blueprint.n_depots):
            pos = get_depot_position(blueprint, rng)
            depot_positions.append(pos)

        customer_position = get_customer_position(blueprint, rng)
        original_locations = deepcopy(depot_positions)
        for pos in customer_position:
            original_locations.append(pos)

        original_locations = np.array(original_locations)

        demand = get_customer_demand(blueprint, customer_position, rng)
        #demand = np.insert(demand, 0, 0, axis=0)

        if blueprint.grid_size == 1000:
            locations = original_locations / 1000
        elif blueprint.grid_size == 1000000:
            locations = original_locations / 1000000
        else:
            assert blueprint.grid_size == 1
            locations = original_locations
        depot_indices = list(range(blueprint.n_depots))

        mdvrp_instance = MDVRPInstance(
                depot_indices       = depot_indices,
                locations           = locations, 
                original_locations  = original_locations,
                demand              = demand, 
                capacity            = blueprint.capacity, 
                use_cost_memory     = use_cost_memory,
                )
        return mdvrp_instance


def get_depot_position(blueprint, rng):
    if blueprint.depot_position == 'R':
        if blueprint.grid_size == 1:
            return rng.uniform(size=2)
        elif blueprint.grid_size == 1000:
            return rng.integers(0, 1001, 2)
        elif blueprint.grid_size == 1000000:
            return rng.integers(0, 1000001, 2)
    elif blueprint.depot_position == 'C':
        if blueprint.grid_size == 1:
            return np.array([0.5, 0.5])
        elif blueprint.grid_size == 1000:
            return np.array([500, 500])
    elif blueprint.depot_position == 'E':
        return np.array([0, 0])
    else:
        raise Exception("Unknown depot position")


#def get_customer_position_clustered(nb_customers, blueprint, rng):
#    assert blueprint.grid_size == 1000
#    random_centers = rng.integers(0, 1001, (blueprint.nb_customers_cluster, 2))
#    customer_positions = []
#    while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
#        random_point = rng.integers(0, 1001, (1, 2))
#        a = random_centers
#        b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
#        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
#        acceptance_prob = np.sum(np.exp(-distances / 40))
#        if acceptance_prob > rng.random():
#            customer_positions.append(random_point[0])
#    return np.concatenate((random_centers, np.array(customer_positions)), axis=0)

def get_customer_position_clustered(nb_customers, blueprint, rng):
    if blueprint.grid_size >= 1000:
        random_centers = rng.integers(0, blueprint.grid_size+1, (blueprint.nb_customers_cluster, 2))
        customer_positions = []
        while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
            random_point = rng.integers(0, blueprint.grid_size+1, (1, 2))
            a = random_centers
            b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
            distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
            acceptance_prob = np.sum(np.exp(-distances / (0.04*blueprint.grid_size)))
            if acceptance_prob > rng.random():
               customer_positions.append(random_point[0])

    elif blueprint.grid_size == 1:
        random_centers = rng.random(size=(blueprint.nb_customers_cluster, 2))
        customer_positions = []
        while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
            random_point = rng.random(size=(1, 2))
            a = random_centers
            b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
            distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
            acceptance_prob = np.sum(np.exp(-distances / (0.04*blueprint.grid_size)))
            if acceptance_prob > rng.random():
                customer_positions.append(random_point[0])
    else:
        raise ValueError(f"blueprint.grid_size is neither >=1000 nor 1")

    return np.concatenate((random_centers, np.array(customer_positions)), axis=0)



def get_customer_position(blueprint, rng):
    if blueprint.customer_position == 'R':
        if blueprint.grid_size == 1:
            return rng.uniform(size=(blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000:
            return rng.integers(0, 1001, (blueprint.nb_customers, 2))
        elif blueprint.grid_size == 1000000:
            return rng.integers(0, 1000001, (blueprint.nb_customers, 2))
    elif blueprint.customer_position == 'C':
        return get_customer_position_clustered(blueprint.nb_customers, blueprint, rng)
    elif blueprint.customer_position == 'RC':
        customer_position = get_customer_position_clustered(int(blueprint.nb_customers / 2), blueprint, rng)
        customer_position_2 = rng.integers(0, 1001, (blueprint.nb_customers - len(customer_position), 2))
        return np.concatenate((customer_position, customer_position_2), axis=0)


def get_customer_demand(blueprint, customer_position, rng):
    if blueprint.demand_type == 'inter':
        if blueprint.n_depots == 1:
            return rng.integers(blueprint.demand_min, blueprint.demand_max + 1, size=blueprint.nb_customers)
        elif blueprint.n_depots > 1:
            demands = rng.integers(blueprint.demand_min, blueprint.demand_max + 1, size=blueprint.nb_customers + blueprint.n_depots)
            depot_indices = list(range(blueprint.n_depots))
            demands[depot_indices] = 0
            return demands
    elif blueprint.demand_type == 'U':
        return np.ones(blueprint.nb_customers, dtype=int)
    elif blueprint.demand_type == 'SL':
        small_demands_nb = int(rng.uniform(0.7, 0.95, 1).item() * blueprint.nb_customers)
        demands_small = rng.integers(1, 11, size=small_demands_nb)
        demands_large = rng.integers(50, 101, size=blueprint.nb_customers - small_demands_nb)
        demands = np.concatenate((demands_small, demands_large), axis=0)
        rng.shuffle(demands)
        return demands
    elif blueprint.demand_type == 'Q':
        assert blueprint.grid_size == 1000
        demands = np.zeros(blueprint.nb_customers, dtype=int)
        for i in range(blueprint.nb_customers):
            if (customer_position[i][0] > 500 and customer_position[i][1] > 500) or (
                    customer_position[i][0] < 500 and customer_position[i][1] < 500):
                demands[i] = rng.integers(51, 101, 1).item()
            else:
                demands[i] = rng.integers(1, 51, 1).item()
        return demands
    elif blueprint.demand_type == 'minOrMax':
        demands_small = np.repeat(blueprint.demand_min, blueprint.nb_customers * 0.5)
        demands_large = np.repeat(blueprint.demand_max, blueprint.nb_customers - (blueprint.nb_customers * 0.5))
        demands = np.concatenate((demands_small, demands_large), axis=0)
        rng.shuffle(demands)
        return demands
    else:
        raise Exception("Unknown customer demand.")


def read_instance(path, pkl_instance_idx=0):
    if path.endswith('.vrp'):
        return read_instance_vrp(path)
    elif path.endswith('.mdvrp'):
        return read_instance_mdvrp(path) 
    elif path.endswith('.sd'):
        return read_instance_sd(path)
    elif path.endswith('.pkl'):
        return read_instances_pkl(path, pkl_instance_idx, 1)[0]
    else:
        raise Exception("Unknown instance file type.")

def read_instance_mdvrp(path) -> MDVRPInstance:
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith("NUM_DEPOTS"):
            num_depots = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
            #quick check on customers ids so that they follow 'the start from 1 convention'
            assert locations[0, 0] == 1, f"Error in reading {path}. Node indices start from {locations[0, 0]}, expected 1"
            locations = np.insert(locations, 0, [0, np.nan, np.nan], axis=0)
    
            locations = locations[:, 1:] # drop ids 

            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
            demand = np.insert(demand, 0, [0, np.nan], axis=0)
            i = i + dimension
        elif line.startswith('TYPE'):
            problem_type = line.split(':')[1]
            problem_type = line.split(' ')[-1]
            assert problem_type == "MDVRP", f"Wrong problem type in data: passed: {problem_type}"
        elif line.startswith('DEPOT_SECTION'):
            depot_indices = np.loadtxt(lines[i + 1:i + 1 + num_depots], dtype=int)
            i = i + num_depots 
        i += 1


    if ((locations[1:,:] > 0) & (locations[1:,:] < 1)).all():
        #then grid size is 0, 1
        grid_size = 1
    elif ((locations[1:,:] > 0) & (locations[1:,:] < 1000)).all():
        #then grid size is likely 1000
        grid_size = 1000
    elif ((locations[1:,:] > 0) & (locations[1:,:] < 1000000)).all():
        #then grid size is likely 1000000
        grid_size = 1000000
    else:
        raise ValueError(f"Error in estimating grid size")
    

#    original_locations = locations[:, 1:]
    original_locations = locations
    locations = original_locations / grid_size 
    demand = demand[:, 1:].squeeze()
    #depot_indices = depot_indices[:, 1:].squeeze()
    depot_indices = depot_indices.tolist()
    if min(depot_indices) != 0:
        depot_indices = [idx-1 for idx in depot_indices]
    assert min(depot_indices) == 0, f"Error in depot_indices convention"

    instance = MDVRPInstance(
            depot_indices = depot_indices,
            locations = locations,
            original_locations = original_locations,
            demand = demand, 
            capacity = capacity,
            )

    return instance

def read_instance_vrp(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / 1000
    demand = demand[:, 1:].squeeze()

    instance = VRPInstance(dimension - 1, locations, original_locations, demand, capacity)
    return instance


def read_instance_sd(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    locations = original_locations / (original_locations[0, 0] * 2)
    demand = demand[:, 1:].squeeze()

    instance = VRPInstance(dimension - 1, locations, original_locations, demand, capacity)
    return instance


#def read_instances_pkl(path, problem_type='vrp', offset=0, num_samples=None):
#
#    instances = []
#
#    with open(path, 'rb') as f:
#        data = pickle.load(f)
#
#    if num_samples is None:
#        num_samples = len(data)
#
#    for args in data[offset:offset + num_samples]:
#        depot, loc, demand, capacity, *args = args
#        loc.insert(0, depot)
#        demand.insert(0, 0)
#
#        locations = np.array(loc)
#        demand = np.array(demand)
#        if problem_type=='vrp':
#            instance = VRPInstance(len(loc) - 1, locations, locations, demand, capacity)
#        else:
#            instance = MDVRPInstance(len(loc) - 1, locations, locations, demand, capacity)
#        instances.append(instance)
#
#    return instances

def read_instances_pkl(path, offset=0, num_samples=None):

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if data and isinstance(data[0], (VRPInstance, MDVRPInstance)):
        return data[offset:offset + (num_samples or len(data))]

    instances = []
    if num_samples is None:
        num_samples = len(data)
    
    for idx, rec in enumerate(data[offset:offset + num_samples], start=offset):
        if not (isinstance(rec, (list, tuple)) and len(rec) == 4):
            raise ValueError(
                f"Record #{idx}: expected 4-tuple (depots, loc, demand, capacity), got \
                type/len={type(rec)}/{len(rec) if hasattr(rec, '__len__') else 'n/a'}"
            )

        depots_raw, loc, demand_raw, capacity = rec
        
        #normalize depots 
        depots = np.asarray(depots_raw, dtype=float)
        if depots.ndim == 1:
            #single depot (2, ) -> (1,2)
            if depots.shape[0] !=2:
                raise ValueError(f"Record #{idx}: single depot must be shape (2,), got {depots.shape}.")
            depots = depots[None, :]
        elif depots.ndim == 2:
            if depots.shape[1] != 2:
                raise ValueError(f"Record #{idx}: depots must have shape (D,2), got {depots.shape}.")
        else:
            raise ValueError(f"Record #{idx}: depots must be (2,) or (D,2), got {depots.shape}.")

        #normalize customers to (N,2)
        cust_loc = np.asarray(loc, dtype=float)
        if cust_loc.ndim != 2 or cust_loc.shape[1] != 2:
            raise ValueError(f"Record #{idx}: customer locations must be (N,2), got {cust_loc.shape}.")
        
        #normalize demand to (N,) and validate length
        demand = np.asarray(demand_raw, dtype=int).ravel()
        N = cust_loc.shape[0]
        if demand.shape[0] != N:
            raise ValueError(
               f"Record #{idx}: demand length {demand.shape[0]} != number of customers {N}."
            )
        

        D = depots.shape[0]
        locations = np.vstack([depots, cust_loc])
        demand_full = np.concatenate([np.zeros(D, dtype=int), demand])
        depot_indices = list(range(D))
    
        if len(demand_full) != D+N:
            raise ValueError(
                f"Problem in shape of demand_full: Expected {D+N} but got {len(demand_full)}"
            )

        instance = MDVRPInstance(
            depot_indices       = depot_indices,
            locations           = locations,
            original_locations  = locations,
            demand              = demand_full,
            capacity            = int(capacity),
        )
        instances.append(instance)

    return instances


def read_vrplib_to_mdvrpinstance(instance_path):
    """
    Reads a VRPLIB formatted instance and returns a MDVRPInstance object
    """
    file = open(instance_path, 'r')
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith("NUM_DEPOTS"):
            num_depots = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension
        elif line.startswith('TYPE'):
            problem_type = line.split(':')[1]
            problem_type = line.split(' ')[-1]
            assert problem_type == "MDVRP", f"Wrong problem type in data: passed: {problem_type}"
        elif line.startswith('DEPOT_SECTION'):
            depot_indices = np.loadtxt(lines[i + 1:i + 1 + num_depots], dtype=int)
            i = i + num_depots 
        i += 1

    locations[:, 0] = locations[:, 0] + 1
    demand[:, 0] = demand[:, 0] + 1

    original_locations = locations[:, 1:] 
    locations = original_locations / 1000
    demand = demand[:, 1:].squeeze()
    depot_indices = depot_indices[:, 1:].squeeze()
    depot_indices = depot_indices.tolist()

    instance = MDVRPInstance(
            depot_indices = depot_indices,
            locations = locations,
            original_locations = original_locations,
            demand = demand, 
            capacity = capacity)
    return instance


def save_dataset_pkl(instances, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(instances, f)
    print(f"Saved dataset to {output_path}")

def _coords_to_vrplib_ints(original_locations):     # scale coords to integers
    arr = np.asarray(original_locations, dtype=float)
    maxv = float(np.max(arr))
    if maxv <= 1.0 + 1e-12:
        arr = np.rint(arr * 1000.0)
    return arr.astype(int)

def _write_vrp_instance(path, inst):
    coords = _coords_to_vrplib_ints(inst.original_locations)
    demand = np.asarray(inst.demand, dtype=int).ravel()
    dim = coords.shape[0]
    cap = int(inst.capacity)

    with open(path, "w") as f:
        name = os.path.splitext(os.path.basename(path))[0]
        f.write(f"NAME : {name}\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {dim}\n")
        f.write(f"CAPACITY : {cap}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i} {int(x)} {int(y)}\n")
        f.write("DEMAND_SECTION\n")
        for i, d in enumerate(demand):
            f.write(f"{i} {int(d)}\n")
        f.write("EOF\n")

def _write_mdvrp_instance(path, inst):
    coords = _coords_to_vrplib_ints(inst.original_locations)
    demand = np.asarray(inst.demand, dtype=int).ravel()
    dim = coords.shape[0]
    cap = int(inst.capacity)

    depot_indices = list(map(int, inst.depot_indices))
    num_depots = len(depot_indices)
    if min(depot_indices) == 0:
        shift = 1
    else:
        shift = 0

    with open(path, "w") as f:
        name = os.path.splitext(os.path.basename(path))[0]
        f.write(f"NAME : {name}\n")
        f.write("TYPE : MDVRP\n")  # required by read_instance_mdvrp()
        f.write(f"DIMENSION : {dim}\n")
        f.write(f"CAPACITY : {cap}\n")
        f.write(f"NUM_DEPOTS : {num_depots}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i+1} {int(x)} {int(y)}\n")
        f.write("DEMAND_SECTION\n")
        for i, d in enumerate(demand):
            f.write(f"{i+1} {int(d)}\n")
        f.write("DEPOT_SECTION\n")
        for row_id, dep_idx in enumerate(depot_indices):
            #f.write(f"{row_id + 1} {dep_idx + shift}\n")
            f.write(f"{row_id + 1}\n")
        f.write("EOF\n")

def save_dataset_vrplib(
    instances,
    folder: str,
    prefix: str = "inst",
    start_index: int = 0
):
    assert (isinstance(instances[0], MDVRPInstance)) or (isinstance(instances[0], VRPInstance))
    os.makedirs(folder, exist_ok=True)

    for k, inst in enumerate(instances, start=start_index):
        if isinstance(inst, MDVRPInstance):
            out_path = os.path.join(folder, f"{prefix}_{k:05d}.mdvrp")
            _write_mdvrp_instance(out_path, inst)
        elif isinstance(inst, VRPInstance):
            out_path = os.path.join(folder, f"{prefix}_{k:05d}.vrp")
            _write_vrp_instance(out_path, inst)
        else:
            raise TypeError(
                f"Unsupported instance type at index {k}: {type(inst).__name__}. "
                "Expected VRPInstance or MDVRPInstance."
            )
    print(f"Saved dataset to {folder}")

#def NLNS_ins_to_pyvrp_sol(instance_path: str, final_instance: MDVRPInstance):
#    if not instance_path or not os.path.isfile(instance_path):
#        raise SystemExit("Provide a valid --instance_path to a VRPLIB file.")
#    problem_data = pyvrp_read(instance_path)
#    print(f"DEBUG: read problem_data: {problem_data}")
#    
#    routes = []
#    for route in final_instance.solution:
#        visits = [el[0] for el in route] 
#        #routes.append(Route(data=problem_data, visits=visits, vehicle_type=0)) 
#        routes.append(visits) 
#
#    print(f"DEBUG: routes: {routes}")
#
#    sol = Solution(
#        data=problem_data,
#        routes=routes
#        )
#
#    return sol

class PlotRoute:
    """
    A lightweight route: a sequence of client indices (no depots inside),
    plus the start/end depot indices. Created to be plottable by plot_solution of PyVRP.
    """
    def __init__(self, visits: Iterable[int], start_depot: int, end_depot: int):
        self._visits: List[int] = [int(v) for v in visits]   # clients only
        self._start = int(start_depot)
        self._end = int(end_depot)

        if len(self._visits) == 0:
            raise ValueError("PlotRoute requires at least one client for plot_solution().")

    # --- what plot_solution needs ---
    def __len__(self) -> int:
        return len(self._visits)

    def __iter__(self):
        return iter(self._visits)

    def __getitem__(self, i: int) -> int:
        return self._visits[i]

    def __array__(self, dtype=None):
        # lets NumPy treat `route` as an index array: x_coords[route]
        arr = np.asarray(self._visits, dtype=np.intp)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def start_depot(self) -> int:
        return self._start

    def end_depot(self) -> int:
        return self._end


class PlotSolution:
    """
    Minimal "solution" wrapper exposing .routes() -> iterable[PlotRoute]. Created to be plottable by plot_solution of PyVRP.
    """
    def __init__(self, routes: Iterable[PlotRoute]):
        self._routes = list(routes)

    def routes(self) -> Iterable[PlotRoute]:
        return self._routes


def mdvrp_to_plot_solution(inst) -> PlotSolution:
    """
    Converts MDVRPInstance.solution (list of tours of [node, demand, nn_idx])
    into a PlotSolution that plot_solution() can draw.

    Keeps only complete tours that start and end at a depot and have >= 1 client.
    """
    assert inst.solution is not None, "Instance has no solution to plot."

    routes: List[PlotRoute] = []
    depot_set = set(inst.depot_indices)

    for tour in inst.solution:
        # Skip depot placeholders or incomplete tours
        if len(tour) < 3:
            continue
        start, end = tour[0][0], tour[-1][0]
        if start not in depot_set or end not in depot_set:
            # Incomplete: skip (or handle specially if you want to visualize partial routes)
            continue

        # Middle of the tour should be only clients; still filter out any accidental depots
        visits = [node for (node, _, _) in tour[1:-1] if node not in depot_set]
        if visits:
            routes.append(PlotRoute(visits=visits, start_depot=start, end_depot=end))

    return PlotSolution(routes)
