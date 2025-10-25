import argparse
import os
import numpy as np
from vrp.data_utils import generate_Instance, get_blueprint

def write_mdvrplib(filename, config, name="problem", convention='mine'):
    """ Generates instances of mdvrp based on 'generate_data.py' of DeepMDV repo.
        Returns them in the VRPLIB format as 'write_vrplib' below """

    blueprint = get_blueprint(config['instance_blueprint'])
    assert config['grid_size'] == 1000 or config['grid_size'] == 1000000
    assert convention in ['mine', 'vrplib']

    MIN_LAT = 0
    MAX_LAT = blueprint.grid_size
    MIN_LON = 0
    MAX_LON = blueprint.grid_size

    if convention == 'mine':
        shift = 0
    else:
        shift = 1

    depot_indices = list(range(shift, blueprint.n_depots+shift))
    node_latitudes = np.random.randint(low=MIN_LAT, high=MAX_LAT, size=blueprint.nb_customers + blueprint.n_depots).tolist()
    node_longitudes = np.random.randint(low=MIN_LON, high=MAX_LON, size=blueprint.nb_customers + blueprint.n_depots).tolist()
    nodes_coordinates = list(zip(node_latitudes, node_longitudes))
    demands = rng.poisson(lam=1.0, size=blueprint.nb_customers + blueprint.n_depots)
    # Replace any zeros with 1 (to avoid zero demand)
    demands = np.where(demands == 0, 1, demands).astype(float)
    shifted_depot_indices = [el -1 for el in depot_indices]
    demands[shifted_depot_indices] = 0.0
    demands = demands.tolist()

    with open(filename, 'w+') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "MDVRP"),
                ("DIMENSION", blueprint.nb_customers + blueprint.n_depots),
                ("CAPACITY", blueprint.capacity),
                ("NUM_DEPOTS", blueprint.n_depots),
                ("EDGE_WEIGHT_TYPE", config['dist_type']),
                ("VEHICLES", 'INF'),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{} {} {}".format(i + shift, x, y)
            #"{}\t{}\t{}".format(i, x, y)
            for i, (x, y) in enumerate(nodes_coordinates)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{} {}".format(i + shift, int(d))
            #"{}\t{}".format(i, d)
            for i, d in enumerate(demands)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("\n".join([
            "{}".format(d)
            #"{}\t{}".format(i + shift, d)
            #"{}\t{}".format(i, d)
            for i, d in enumerate(depot_indices)
        ]))
        f.write("\n")
        #f.write("-1\n")
        f.write("EOF\n")
        print(f"Written data to file: {filename}")

def write_vrplib(filename, loc, demand, capacity, grid_size, name="problem", convention='mine'):
    assert grid_size == 1000 or grid_size == 1000000
    if convention == 'mine':
        shift = 0
    else:
        shift = 1

    depot_indices = [1]

    with open(filename, 'w+') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc)),
                ("CAPACITY", capacity),
                ("NUM_DEPOTS", 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("VEHICLES", 'INF'),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{} {} {}".format(i + 1, x, y)
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate(demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
#        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default='vrp',
            help="Problem: 'vrp', 'mdvrp'")
    parser.add_argument("--data_dir", default='data')
    parser.add_argument('--instance_blueprint', default=None, type=str)
    parser.add_argument('--dataset_size', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--grid_size', default=1000, type=int)
    parser.add_argument('--vrp_size', type=int) #number of customers
    parser.add_argument('--depot_size', type=int)
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--dist_type', type=str, default='EUC_2D')
    parser.add_argument('--convention', type=str, default='mine')

    config = parser.parse_args()

    rng = np.random.default_rng(config.seed)
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    if config.problem == "mdvrp":
        for i in range(config.dataset_size):
            name = "mdvrp_seed_{}_id_{}".format(config.seed, i)
            filename = os.path.join(config.data_dir, name + ".mdvrp")
            write_mdvrplib(filename, vars(config), name, config.convention)

    elif config.problem == "vrp":
        assert config.instance_blueprint.startswith("XE_"), 'Only XE are supported'
        blueprint = get_blueprint(config.instance_blueprint)

        for i in range(config.dataset_size):
            instance = generate_Instance(blueprint=blueprint, use_cost_memory=False, rng=rng)

            name = "{}_seed_{}_id_{}".format(config.instance_blueprint, config.seed, i)
            filename = os.path.join(config.data_dir, name + ".vrp")
            write_vrplib(filename, instance.original_locations, instance.demand, instance.capacity,
                         blueprint.grid_size, name)
