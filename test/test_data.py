import unittest
from generate_data import generate_mdvrp_data

class TestMDVRP(unittest.TestCase):
    def setUp(self):
        self.geodata_file = "/home/pettepiero/tirocinio/NLNS/test/test_data_geo.csv"
        self.config = { 
            'dataset_size': 1, 
            'vrp_size': 20, 
            'depot_size': 3,
            'capacity': 50
        }

    def test_generate_mdvrp_data(self):

        dataset = generate_mdvrp_data(
                geodata_file = self.geodata_file, 
                config = self.config)

        print(dataset)

