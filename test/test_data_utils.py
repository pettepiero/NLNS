import json
import unittest
import numpy as np
import argparse
from vrp.data_utils import *
SEED=0

class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)

class InstanceBlueprint_test(unittest.TestCase):

    def setUp(self):
        self.tw_options = {'tw_min': 0, 'tw_max': 1000, 'avg_window': 30}
        self.blueprint_config = {
                'problem_type': 'mdvrptw',
                'nb_customers': 20,
                'depot_position': 'R',
                'customer_position': 'R',
                'nb_customer_cluster': 2,
                'demand_type': 'inter',
                'demand_min': 1,
                'demand_max': 5,
                'capacity': 50,
                'grid_size': 1,
                'n_depots': 3,
                'tw_options': self.tw_options,
            }
        self.rng = np.random.default_rng(SEED)

        with open("config_example.json", 'r') as f:
            config = json.load(f)
        self.config = argparse.Namespace(**config)
        

    def test_generate_blueprint(self):
        self.blueprint = InstanceBlueprint(**self.blueprint_config)
        b = self.blueprint
        #assertEqual values
        self.assertEqual(b.problem_type, 'mdvrptw')
        self.assertEqual(b.nb_customers, 20)
        self.assertEqual(b.depot_position, 'R')
        self.assertEqual(b.customer_position, 'R')
        self.assertEqual(b.nb_customers_cluster, 2)
        self.assertEqual(b.demand_type, 'inter')
        self.assertEqual(b.demand_min, 1)
        self.assertEqual(b.demand_max, 5)
        self.assertEqual(b.capacity, 50)
        self.assertEqual(b.grid_size, 1)
        self.assertEqual(b.n_depots, 3)
        self.assertEqual(b.tw_options, self.tw_options)


    def test_get_customer_time_windows(self):
        self.blueprint = InstanceBlueprint(**self.blueprint_config)
        time_windows = get_customer_time_windows(self.blueprint, self.rng)
        self.assertEqual(len(time_windows), self.blueprint.nb_customers + self.blueprint.n_depots) 


    def test_create_dataset_mdvrptw(self):
        dataset = create_dataset(size=self.config.batch_size, config=self.config, create_solution=False, use_cost_memory=True, seed=self.config.seed) 

        self.assertEqual(len(dataset), self.config.batch_size)
        print(dataset[0].solution)

    def test_create_initial_solution(self):
        dataset = create_dataset(size=self.config.batch_size, config=self.config, create_solution=True, use_cost_memory=True, seed=self.config.seed) 
        for el in dataset[0].solution:
            print(el)
