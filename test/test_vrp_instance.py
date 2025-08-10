import unittest
from vrp.mdvrp_problem import MDVRPInstance


class MDVRP_instance_test(unittest.TestCase):
    def setUp(self):
        self.depot_indices = [0, 1, 2, 3]
        self.locations = [[0.53133224, 0.28243662],
                     [0.6743981 , 0.64041828],
                     [0.4113433 , 0.318311  ],
                     [0.94992566, 0.03165872],
                     [0.63494319, 0.20385504],
                     [0.10591025, 0.9292119 ],
                     [0.32258949, 0.16922623],
                     [0.93441326, 0.50766116],
                     [0.46652433, 0.22617598],
                     [0.37048569, 0.05035837],
                     [0.24523844, 0.67245775],
                     [0.23323873, 0.088866  ],
                     [0.47995916, 0.17653291],
                     [0.83341379, 0.10344553],
                     [0.35924599, 0.03583394],
                     [0.61682411, 0.01871345],
                     [0.82985747, 0.50048258],
                     [0.92230851, 0.46465877],
                     [0.82022165, 0.05927801],
                     [0.65409717, 0.45940834],
                     [0.2664566 , 0.76469668]]
        self.original_locations = self.locations
        self.demand = [0, 0, 0, 0, 1, 5, 6, 6, 1, 2, 2, 9, 9, 3, 5, 1, 4, 9, 3, 3, 9] 
        self.capacity = 50

        # assert on initial data
        self.assertEqual(len(self.demand), len(self.locations))

        for i in self.depot_indices:
            self.assertEqual(self.demand[i], 0)

    
    def test_constructor(self):
        mdvrp_instance = MDVRPInstance(
                depot_indices       = self.depot_indices,
                locations           = self.locations,
                original_locations  = self.locations,
                demand              = self.demand,
                capacity            = self.capacity,
        )

        self.assertEqual(mdvrp_instance.depot_indices, self.depot_indices)
        self.assertEqual(mdvrp_instance.locations, self.locations)
        self.assertEqual(mdvrp_instance.original_locations, self.original_locations)
        self.assertEqual(mdvrp_instance.demand, self.demand)
        self.assertEqual(mdvrp_instance.capacity, self.capacity)

                
