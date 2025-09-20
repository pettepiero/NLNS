import unittest
from vrp.data_utils import save_dataset_pkl, read_instances_pkl, read_instance_mdvrp
from pathlib import Path
import os


class TestRead_Write_NOOP(unittest.TestCase):
    def test_noop(self):
        directory = Path('./test_subdataset/')
        inst_list = os.listdir(directory)
        inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.mdvrp']

        initial_dataset = []
        for inst in inst_list:
            instance = read_instance_mdvrp(os.path.join(directory, inst))
            initial_dataset.append(instance)
        save_dataset_pkl(initial_dataset, './test/test_dataset.pkl')
        
        read_dataset = read_instances_pkl('./test/test_dataset.pkl')

        
        self.assertEqual(len(initial_dataset), len(read_dataset))
        for i, inst in enumerate(initial_dataset):

            self.assertEqual(initial_dataset[i], read_dataset[i])

        
