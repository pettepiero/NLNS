import unittest
from vrp.data_utils import save_dataset_pkl, read_instances_pkl, read_instance_mdvrp
from pathlib import Path
import os
import numpy as np

class TestRead_Write_NOOP(unittest.TestCase):
    def test_noop(self):
        directory = Path('./test_subdataset/')
        inst_list = os.listdir(directory)
        inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.mdvrp']

        initial_dataset = [read_instance_mdvrp(os.path.join(directory, inst)) for inst in inst_list]
        #for inst in inst_list:
        #    instance = read_instance_mdvrp(os.path.join(directory, inst))
        #    initial_dataset.append(instance)
        save_dataset_pkl(initial_dataset, './test/test_dataset.pkl')
        
        read_dataset = read_instances_pkl('./test/test_dataset.pkl')

        print(f"Testing the following variables:")
        for el in vars(read_dataset[0]):
            print(f"\t{el}")


        self.assertEqual(len(initial_dataset), len(read_dataset))
        for i in range(len(initial_dataset)):
            self.assertEqual(initial_dataset[i].depot_indices, read_dataset[i].depot_indices)
            self.assertEqual(initial_dataset[i].customer_indices, read_dataset[i].customer_indices)
            self.assertTrue(
                    np.array_equal(
                        initial_dataset[i].locations, 
                        read_dataset[i].locations,
                        equal_nan=True), 
                    f"Failed on iteration {i}")
            self.assertTrue(
                    np.array_equal(
                        initial_dataset[i].original_locations, 
                        read_dataset[i].original_locations,
                        equal_nan=True), 
                    f"Failed on iteration {i}")
            self.assertTrue(
                    np.array_equal(
                        initial_dataset[i].demand, 
                        read_dataset[i].demand,
                        equal_nan=True), 
                    f"Failed on iteration {i}")
            self.assertEqual(initial_dataset[i].capacity, read_dataset[i].capacity)
            self.assertEqual(initial_dataset[i].solution, read_dataset[i].solution)
            self.assertEqual(initial_dataset[i].nn_input_idx_to_tour, read_dataset[i].nn_input_idx_to_tour)
            self.assertEqual(initial_dataset[i].open_nn_input_idx, read_dataset[i].open_nn_input_idx)
            self.assertEqual(initial_dataset[i].incomplete_tours, read_dataset[i].incomplete_tours)
            self.assertTrue(
                    np.array_equal(
                        initial_dataset[i].costs_memory, 
                        read_dataset[i].costs_memory,
                        equal_nan=True), 
                    f"Failed on iteration {i}")
        self.maxDiff = None

        os.remove('./test/test_dataset.pkl')
        
