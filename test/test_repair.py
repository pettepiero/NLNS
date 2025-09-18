import unittest
from vrp.mdvrp_problem import MDVRPInstance
from repair import get_depot_mask
import numpy as np
from vrp.data_utils import create_dataset
from actor import VrpActorModel
import argparse
import torch

SEED = 0
BATCH_SIZE = 15
BLUEPRINT = 'MD_7'

class Repair_test(unittest.TestCase):
    def setUp(self):
        # create instances
        self.rng = np.random.default_rng(SEED)
        self.batch_size = BATCH_SIZE 
        
        #create config
        self.config = argparse.Namespace(
                instance_blueprint=BLUEPRINT, 
                problem_type='mdvrp',
                seed=SEED,
                capacity=6,
                device='cpu',
                split_delivery=False,
                )

        # generate batch_size instances
        self.training_set = create_dataset(size=self.batch_size, config=self.config,
                                  create_solution=True, use_cost_memory=False, seed=self.config.seed)
        for instance in self.training_set:
            instance.destroy_point_based(p=0.3, rng=self.rng)
        self.n_points = max([ins.get_max_nb_input_points() for ins in self.training_set])

        # create actor
#        self.actor = VrpActorModel(self.config.device, hidden_size=self.config.pointer_hidden_size).to(self.config.device    )
#        self.actor.train()

    def test_get_depot_mask(self):
        depot_mask = get_depot_mask(
            batch_size  = self.batch_size,
            n_points    = self.n_points,
            config      = self.config,
            instances   = self.training_set,
            ) 
        known_sol = torch.zeros((self.batch_size, self.n_points), dtype=torch.bool, device=self.config.device)
        for i, ins in enumerate(self.training_set):
            known_sol[i, :ins.n_depots] = True
            self.assertTrue(np.array_equal(np.array(known_sol[i]), np.array(depot_mask[i])))

    def test__actor_model_forward(self):
        # test ptr

        # test logp
