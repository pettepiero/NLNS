import numpy as np
from vrp.vrp_problem import VRPInstance
from vrp.mdvrp_problem import MDVRPInstance
import pickle
from tqdm import trange 

class InstanceBlueprint:
    """Describes the properties of a certain instance type (e.g. number of customers)."""

    def __init__(self, nb_customers, depot_position, customer_position, nb_customer_cluster, demand_type, demand_min, demand_max, capacity, grid_size, n_depots=1):
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
        assert config.instance_blueprint in ['MD_1', 'MD_2', 'MD_3', 'MD_4', 'MD_5', 'MD_6']

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
        original_locations = depot_positions
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
            return rng.uniform(size=(1, 2))
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


def get_customer_position_clustered(nb_customers, blueprint, rng):
    assert blueprint.grid_size == 1000
    random_centers = rng.integers(0, 1001, (blueprint.nb_customers_cluster, 2))
    customer_positions = []
    while len(customer_positions) + blueprint.nb_customers_cluster < nb_customers:
        random_point = rng.integers(0, 1001, (1, 2))
        a = random_centers
        b = np.repeat(random_point, blueprint.nb_customers_cluster, axis=0)
        distances = np.sqrt(np.sum((a - b) ** 2, axis=1))
        acceptance_prob = np.sum(np.exp(-distances / 40))
        if acceptance_prob > rng.random():
            customer_positions.append(random_point[0])
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

def read_instance_mdvrp(path):
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
    depot_indices = depot_indices.squeeze().tolist()

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


def read_instances_pkl(path, offset=0, num_samples=None):
    instances = []

    with open(path, 'rb') as f:
        data = pickle.load(f)

    if num_samples is None:
        num_samples = len(data)

    for args in data[offset:offset + num_samples]:
        depot, loc, demand, capacity, *args = args
        loc.insert(0, depot)
        demand.insert(0, 0)

        locations = np.array(loc)
        demand = np.array(demand)

        instance = VRPInstance(len(loc) - 1, locations, locations, demand, capacity)
        instances.append(instance)

    return instances

def read_instances_pkl(path, offset=0, num_samples=None):
    instances = []

    with open(path, 'rb') as f:
        data = pickle.load(f)

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
