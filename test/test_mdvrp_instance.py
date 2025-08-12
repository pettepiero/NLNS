import unittest
import numpy as np
from vrp.mdvrp_problem import MDVRPInstance

class MDVRP_instance_test(unittest.TestCase):
    """ Tests based on the instance generated in setUp() method.
        Hand written numbers, see also `locations.ods` file as a reference """
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

    def test_destroy(self):
        self.mdvrp_instance.create_initial_solution()
        customers_to_remove = [4, 5, 14, 19, 6, 15, 17, 18]
        self.mdvrp_instance.destroy(customers_to_remove)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours

        self.assertIn(member=[[4, 1, None]], container=sol)
        self.assertIn(member=[[5, 5, None]], container=sol)
        self.assertIn(member=[[14, 5, None]], container=sol)
        self.assertIn(member=[[19, 3, None]], container=sol)
        self.assertIn(member=[[6, 6, None]], container=sol)
        self.assertIn(member=[[15, 1, None]], container=sol)
        self.assertIn(member=[[17, 9, None]], container=sol)
        self.assertIn(member=[[18, 3, None]], container=sol)

        # test if inserted in incomplete solution too
        self.assertIn(member=[[4, 1, None]], container=incomplete_tours)
        self.assertIn(member=[[5, 5, None]], container=incomplete_tours)
        self.assertIn(member=[[14, 5, None]], container=incomplete_tours)
        self.assertIn(member=[[19, 3, None]], container=incomplete_tours)
        self.assertIn(member=[[6, 6, None]], container=incomplete_tours)
        self.assertIn(member=[[15, 1, None]], container=incomplete_tours)
        self.assertIn(member=[[17, 9, None]], container=incomplete_tours)
        self.assertIn(member=[[18, 3, None]], container=incomplete_tours)

    def test_destroy_point_based(self):
        self.mdvrp_instance.create_initial_solution()

        point = np.array([[0.14365113, 0.5307886 ]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertIn(member=[[10, 2, None]], container=sol)
        self.assertIn(member=[[10, 2, None]], container=incomplete_tours)

        point = np.array([[0.36975687, 0.91092584]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertIn(member=[[20, 9, None]], container=sol)
        self.assertIn(member=[[20, 9, None]], container=incomplete_tours)

        point = np.array([[0.79329399, 0.1709594 ]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertIn(member=[[13, 3, None]], container=sol)
        self.assertIn(member=[[13, 3, None]], container=incomplete_tours)

    def test_get_incomplete_tours(self):
        self.mdvrp_instance.create_initial_solution()
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertEqual(incomplete_tours, None)

        customers_to_remove = [4, 5, 14, 19, 6, 15, 17, 18]
        self.mdvrp_instance.destroy(customers_to_remove)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours

        # test after destroy (code from test_destroy)
        self.assertIn(member=[[4, 1, None]], container=incomplete_tours)
        self.assertIn(member=[[5, 5, None]], container=incomplete_tours)
        self.assertIn(member=[[14, 5, None]], container=incomplete_tours)
        self.assertIn(member=[[19, 3, None]], container=incomplete_tours)
        self.assertIn(member=[[6, 6, None]], container=incomplete_tours)
        self.assertIn(member=[[15, 1, None]], container=incomplete_tours)
        self.assertIn(member=[[17, 9, None]], container=incomplete_tours)
        self.assertIn(member=[[18, 3, None]], container=incomplete_tours)

    def test_get_max_nb_input_points(self):
        self.mdvrp_instance.create_initial_solution()

        # hand made incomplete solutions instances:
        solution = [
                [[0, 0, 0]], #depot 0
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[1, 0, 1]], #depot 1
                [[1, 0, 1], [5, 2, None], [8, 1, None], [1, 0, 1]],
                [[2, 0, 2]], #depot 2
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]], #non_depot_tour_end 6 
                [[3, 0, 3]]] #depot 3

        # hard code incomplete solutions instances:
        incomplete_tours = [
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]]] #non_depot_tour_end 6 

        self.mdvrp_instance.solution = solution 
        self.mdvrp_instance.incomplete_tours = incomplete_tours 
        n_depots = self.mdvrp_instance.n_depots 
        self.assertEqual(n_depots, 4) #debug
        non_depot_tour_ends = 6 #see comments above
        known_max_nb_input_points = n_depots + non_depot_tour_ends
        max_nb_input_points = self.mdvrp_instance.get_max_nb_input_points()
        self.assertEqual(max_nb_input_points, known_max_nb_input_points) 

        # hand made incomplete solutions instances:
        solution = [
                [[0, 0, 0]], #depot 0
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[1, 0, 1]], #depot 1
                [[1, 0, 1], [5, 2, None], [8, 1, None], [1, 0, 1]],
                [[2, 0, 2]], #depot 2
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]], #non_depot_tour_end 6 
                [[3, 0, 3]], #depot 3
                [[3, 0, 3], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 7 
                [[10, 1, None]], #non_depot_tour_end 8 
                [[17, 2, None], [16, 2, None], [3, 0, 3]]] #non_depot_tour_end 9

        # hard code incomplete solutions instances:
        incomplete_tours = [
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]], #non_depot_tour_end 6 
                [[3, 0, 3], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 7 
                [[10, 1, None]], #non_depot_tour_end 8 
                [[17, 2, None], [16, 2, None], [3, 0, 3]]] #non_depot_tour_end 9

        self.mdvrp_instance.solution = solution 
        self.mdvrp_instance.incomplete_tours = incomplete_tours 
        n_depots = self.mdvrp_instance.n_depots 
        self.assertEqual(n_depots, 4) #debug
        non_depot_tour_ends = 9 #see comments above
        known_max_nb_input_points = n_depots + non_depot_tour_ends
        max_nb_input_points = self.mdvrp_instance.get_max_nb_input_points()
        self.assertEqual(max_nb_input_points, known_max_nb_input_points) 

