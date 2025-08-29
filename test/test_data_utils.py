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
        self.tw_options = {'tw_min': 0, 'tw_max': 1000, 'avg_window': 30, 'min_window': 10, 'late_coeff': 10, 'early_coeff': 10}
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
                'speed': 10,
                'grid_size': 1,
                'n_depots': 3,
                'tw_options': self.tw_options,
            }
        self.rng = np.random.default_rng(SEED)

        with open("test/config_example.json", 'r') as f:
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
        self.assertEqual(b.speed, 10)
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

    def test_create_initial_solution(self):
        self.blueprint = get_blueprint(self.config.instance_blueprint)
        instance = generate_Instance(
                blueprint=self.blueprint,
                use_cost_memory=True,
                rng=self.rng) 
        instance.create_initial_solution(self.config, self.tw_options)
        
        self.assertTrue(instance.solution is not None)
        for i in range(instance.n_depots):
            self.assertEqual(instance.solution[i][0][0], instance.depot_indices[i])
            self.assertEqual(instance.solution[i][0][-1], instance.depot_indices[i])

        self.assertEqual(len(instance.solution), len(instance.solution_schedule))

        i = 0
        for sol, sched in zip(instance.solution, instance.solution_schedule):
            self.assertEqual(len(sol), len(sched))

            for j in range(0, len(sol)-1):
                start_idx = sol[j][0]
                end_idx = sol[j+1][0]
                
                self.assertGreater(sched[j+1][0], sched[j][1])

                schedule_time_diff = sched[j+1][0] - sched[j][1]
                time_diff = np.round(instance.distance_matrix[start_idx, end_idx]/instance.speed)
                time_diff_with_window = instance.time_windows[end_idx][0] - sched[j][1]
                # either time diff or time diff + remaining time until start of window
                self.assertTrue((schedule_time_diff == time_diff) or (schedule_time_diff == time_diff_with_window))
            i += 1 
