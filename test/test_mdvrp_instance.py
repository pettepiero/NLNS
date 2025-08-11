import unittest
import numpy as np
from vrp.mdvrp_problem import MDVRPInstance


class MDVRP_instance_test(unittest.TestCase):
    def setUp(self):
        self.depot_indices = [0, 1, 2, 3]
        self.locations = np.array([[0.53133224, 0.28243662],
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
                     [0.2664566 , 0.76469668]])
        self.original_locations = self.locations
        self.demand = [0, 0, 0, 0, 1, 5, 6, 6, 1, 2, 2, 9, 9, 3, 5, 1, 4, 9, 3, 3, 9] 
        self.capacity = 50

        # assert on initial data
        self.assertEqual(len(self.demand), len(self.locations))

        for i in self.depot_indices:
            self.assertEqual(self.demand[i], 0)

        self.mdvrp_instance = MDVRPInstance(
                depot_indices       = self.depot_indices,
                locations           = self.locations,
                original_locations  = self.locations,
                demand              = self.demand,
                capacity            = self.capacity,
        )
    
    def test_constructor(self):
        self.assertEqual(self.mdvrp_instance.depot_indices, self.depot_indices)
        #self.assertEqual(self.mdvrp_instance.locations, self.locations)
        self.assertEqual(np.array_equal(self.mdvrp_instance.locations, self.locations), True)
        self.assertEqual(np.array_equal(self.mdvrp_instance.original_locations, self.original_locations), True)
        self.assertEqual(self.mdvrp_instance.demand, self.demand)
        self.assertEqual(self.mdvrp_instance.capacity, self.capacity)

    def test_get_nearest_depot(self):
        customer_to_depot_known = {
                18: 3, 13: 3, 
                9: 2, 10: 2, 11: 2, 14: 2, 6: 2,
                7: 1, 5: 1, 16: 1, 17: 1, 19: 1, 20: 1,
                4: 0, 8: 0, 12: 0, 15: 0}
        for cust in iter(customer_to_depot_known):
            cust = cust
            result = self.mdvrp_instance.get_nearest_depot(cust)
            known_sol = customer_to_depot_known[cust]
            self.assertEqual(result, known_sol)

    def test_get_n_closest_locations_to(self):
        # some hand made examples:
        closest_locs = {
                18: 13,
                20: 10,
                16: 17,
                12: 8,
                5: 20,
                }
        for loc in iter(closest_locs):
            known_sol = closest_locs[loc]
            mask = np.array([True] * (self.mdvrp_instance.nb_customers + self.mdvrp_instance.n_depots))
            mask[loc] = False
            mask[self.mdvrp_instance.depot_indices] = False
            result = self.mdvrp_instance.get_n_closest_locations_to(loc, mask, 1)[0] # testing first element only for now
            self.assertEqual(result, known_sol, msg=f"Failed on customer {loc}")


    def test_create_initial_solution(self):
        self.mdvrp_instance.create_initial_solution()
        sol = self.mdvrp_instance.solution
        known_solution = [
                [[0, 0, 0]],
                [[0, 0, 0], [8, 1, None], [12, 9, None], [4, 1, None], [15, 1, None], [0, 0, 0]],
                [[1, 0, 1]],
                [[1, 0, 1], [19, 3, None], [16, 4, None], [17, 9, None], [7, 6, None], [20, 9, None], [5, 5, None], [1, 0, 1]],
                [[2, 0, 2]],
                [[2, 0, 2], [6, 6, None], [11, 9, None], [14, 5, None], [9, 2, None], [10, 2, None], [2, 0, 2]],
                [[3, 0, 3]],
                [[3, 0, 3], [18, 3, None], [13, 3, None], [3, 0, 3]]]
        
        self.assertEqual(len(sol), len(known_solution))

        self.assertEqual(sol, known_solution)

