import unittest
import torch
import numpy as np
from vrp.mdvrp_problem import MDVRPInstance, get_mask, get_depot
from vrp.data_utils import create_dataset
from search import destroy_instances
import argparse
from copy import deepcopy

class MDVRP_instance_test(unittest.TestCase):
    """ Tests based on the instance generated in setUp() method.
        Hand written numbers, see also `locations.ods` file as a reference """
    def setUp(self):
        self.depot_indices = [1, 2, 3, 4]
        self.locations = np.array([
                    [None      , None      ],
                    [0.53133224, 0.28243662],
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
        self.demand = [None, 0, 0, 0, 0, 1, 5, 6, 6, 1, 2, 2, 9, 9, 3, 5, 1, 4, 9, 3, 3, 9] 
        self.capacity = 50

        # assert on initial data
        self.assertEqual(len(self.demand), len(self.locations))
        self.assertEqual(self.demand[0], None)
        self.assertTrue(np.array_equal(self.locations[0], [None, None]))

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
                19: 14,
                18: 8,
                11: 21,
                13: 9,
                15: 10,
                }
        for loc in iter(closest_locs):
            known_sol = closest_locs[loc]
            mask = np.array([True] * (self.mdvrp_instance.nb_customers + self.mdvrp_instance.n_depots))
            mask[0] = False
            mask[loc] = False
            mask[self.mdvrp_instance.depot_indices] = False
            result = self.mdvrp_instance.get_n_closest_locations_to(loc, mask, 1)[0] # testing first element only for now
            self.assertEqual(result, known_sol, msg=f"Failed on customer {loc}")


    def test_create_initial_solution(self):
        self.mdvrp_instance.create_initial_solution()
        sol = self.mdvrp_instance.solution
    
        known_solution = [
                [[1, 0, 0]],
                [[2, 0, 1]],
                [[3, 0, 2]],
                [[4, 0, 3]],
                [[1, 0, 0], [9, 1, None], [13, 9, None], [5, 1, None], [16, 1, None], [1, 0, 0]],
                [[2, 0, 1], [20, 3, None], [17, 4, None], [18, 9, None], [8, 6, None], [21, 9, None], [6, 5, None], [2, 0, 1]],
                [[3, 0, 2], [7, 6, None], [12, 9, None], [15, 5, None], [10, 2, None], [11, 2, None], [3, 0, 2]],
                [[4, 0, 3], [19, 3, None], [14, 3, None], [4, 0, 3]]]
        
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
        rng = np.random.default_rng()
        point = np.array([[0.14365113, 0.5307886 ]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point, rng=rng)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertIn(member=[[10, 2, None]], container=sol)
        self.assertIn(member=[[10, 2, None]], container=incomplete_tours)

        point = np.array([[0.36975687, 0.91092584]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point, rng=rng)
        sol = self.mdvrp_instance.solution
        incomplete_tours = self.mdvrp_instance.incomplete_tours
        self.assertIn(member=[[20, 9, None]], container=sol)
        self.assertIn(member=[[20, 9, None]], container=incomplete_tours)

        point = np.array([[0.79329399, 0.1709594 ]])
        p = 0.1
        self.mdvrp_instance.destroy_point_based(p=p, point=point, rng=rng)
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
                [[1, 0, 1]], #depot 1
                [[2, 0, 2]], #depot 2
                [[3, 0, 3]], #depot 3
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[1, 0, 1], [5, 2, None], [8, 1, None], [1, 0, 1]],
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]]] #non_depot_tour_end 6 

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
                [[1, 0, 1]], #depot 1
                [[2, 0, 2]], #depot 2
                [[3, 0, 3]], #depot 3
                [[0, 0, 0], [12, 2, None], [6, 1, None]], #non_depot_tour_end 1 
                [[11, 1, None]], #non_depot_tour_end 2 
                [[7, 2, None], [15, 2, None], [0, 0, 0]], #non_depot_tour_end 3 
                [[1, 0, 1], [5, 2, None], [8, 1, None], [1, 0, 1]],
                [[2, 0, 2], [18, 2, None], [16, 1, None], [19, 2, None]], #non_depot_tour_end 4 
                [[10, 1, None]], #non_depot_tour_end 5 
                [[17, 2, None], [16, 2, None], [2, 0, 2]], #non_depot_tour_end 6 
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

    def test_get_network_input(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)
        incomplete_tours = mdvrp.incomplete_tours
        input_dim = mdvrp.get_max_nb_input_points()

        nn_input_static, nn_input_dynamic = mdvrp.get_network_input(input_dim)
        nn_input = np.concat((nn_input_static, nn_input_dynamic), axis=1)
      
        for idx, v in enumerate(nn_input):
            if idx < mdvrp.n_depots: # testing inputs corresponding to depots
                self.assertEqual(v[0], mdvrp.locations[mdvrp.depot_indices[idx]][0]) #depot coordinates
                self.assertEqual(v[1], mdvrp.locations[mdvrp.depot_indices[idx]][1]) #depot coordinates
                self.assertEqual(-v[2], mdvrp.capacity) #vehicle capacity
                self.assertEqual(v[3], -1) #depot encoding = -1 
            else:
                tour, i = mdvrp.nn_input_idx_to_tour[idx]
                customer = tour[i][0]
                if len(tour)>1:
                    sum_demands = sum([c[1] for c in tour])
                    self.assertEqual(v[0], mdvrp.locations[customer][0]) #customer coordinates
                    self.assertEqual(v[1], mdvrp.locations[customer][1]) #customer coordinates
                    self.assertEqual(v[2], sum_demands) #sum of fulfilled demands
                    if tour[-1][0] in mdvrp.depot_indices:
                        self.assertEqual(v[3], 3) #encoding
                    else:
                        if tour[0][0] in mdvrp.depot_indices:
                            self.assertEqual(v[3], 3) #encoding
                        else:
                            self.assertEqual(v[3], 2) #encoding
                else: #len(tour)=1 -> demand is not sum of demands and encoding is different
                    demand = tour[0][1]
                    self.assertEqual(v[0], mdvrp.locations[customer][0]) #customer coordinates
                    self.assertEqual(v[1], mdvrp.locations[customer][1]) #customer coordinates
                    self.assertEqual(v[2], demand) #sum of fulfilled demands
                    self.assertEqual(v[3], 1) #encoding

        # test mdvrp.open_nn_input_idx
        self.assertEqual(nn_input.shape[0], input_dim)
        
        indices_to_check = []

        for i, el in enumerate(mdvrp.nn_input_idx_to_tour):
            if i < mdvrp.n_depots:
                continue
             
            if len(el[0]) == 1:
                idx = el[0][0][-1]
                indices_to_check.append(idx)
            else:
                #find end
                if el[0][0][0] in mdvrp.depot_indices:
                    idx = el[0][-1][-1]
                    assert idx not in mdvrp.depot_indices, 'Error. A route with starting and ending depot and len > 1 in nn_input_idx_to_tour'
                    indices_to_check.append(idx)
                elif el[0][-1][0] in mdvrp.depot_indices: 
                    idx = el[0][0][-1]
                    assert idx not in mdvrp.depot_indices, 'Error. A route with starting and ending depot and len > 1 in nn_input_idx_to_tour'
                    indices_to_check.append(idx)
                else:
                    raise ValueError

        #reorded list
        #indices_to_check = indices_to_check.sort()
        indices_to_check.sort()
        self.assertEqual(indices_to_check, mdvrp.open_nn_input_idx)
        

                    
                
    def test_do_action(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #0 [[9, 1, 4]] to incomplete tour #4 [[12, 9, 11]]
        # 
        mdvrp.do_action(
            id_from=4, # incomplete tour #0 is index 4 in input vector
            id_to=11) # incomplete tour #7 is index 11 in input vector
        member = [[9, 1, 4], [12, 9, 11]]
        member_reversed = list(reversed(member)) 
        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )

        # connect incomplete tour #4 [[21, 9, 8]] to incomplete tour #6 [[3, 0, 2], [7, 6, 10]]
        # 
        mdvrp.do_action(
            id_from=8, # incomplete tour #4 is index 8 in input vector
            id_to=10) # incomplete tour #6 is index 10 in input vector 
        member = [[3, 0, 2], [7, 6, 10], [21, 9, 8]]
        member_reversed = list(reversed(member)) 

        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )

    def test_do_action2(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(10)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #2 [[5, 1, 6]] to incomplete tour #0 [[1, 0, 0], [9, 1, 4]]
         
        mdvrp.do_action(
            id_from=6, # incomplete tour #2 is index 6 in input vector
            id_to=4) # incomplete tour #0 is index 4 in input vector
        member = [[5, 1, 6], [9, 1, 4], [1, 0, 0]]
        member_reversed = list(reversed(member)) 

        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #2 [[17, 9, 6]] to incomplete tour #0 [[1, 0, 1], [19, 3, 4]]
         
        # connect incomplete tour #3 [[16, 1, 7] to depot [[1, 0, 0]]
         
        mdvrp.do_action(
            id_from=7, # incomplete tour #3 is index 7 in input vector
            id_to=0) # depot 1 is index 0 in input vector 

        member = [[16, 1, 7], [1, 0, 0]]
        member_reversed = list(reversed(member)) 
        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )
        
        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)

        # test that solution doesn't have duplicate customer appearances
        seen = {}
        duplicates = {}

        print(f"DEBUG: mdvrp.solution:")
        for t in mdvrp.solution:
            print(t)
            for node in t:
                cust_id = node[0]
                if cust_id in seen and cust_id not in mdvrp.depot_indices:
                    duplicates.setdefault(cust_id, []).append(t)
                else:
                    seen[cust_id] = t
        print(f"DEBUG: mdvrp.nn_input_idx_to_tour")
        for el in mdvrp.nn_input_idx_to_tour:
            print(el)

        self.assertEqual(len(duplicates), 0, "There are duplicate customers in solution")

        # test position field of each element of nn_input_idx_to_tour
        for el in mdvrp.nn_input_idx_to_tour:
            if len(el[0]) == 1:
                self.assertEqual(el[1], 0)
            else:
                if el[0][0][0] in mdvrp.depot_indices:
                    self.assertEqual(el[1], len(el[0])-1)
                else:
                    self.assertEqual(el[1], 0)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # test the nn_input_idx_to_tour
        for i, el in enumerate(mdvrp.nn_input_idx_to_tour):
            #i-th element of nn_input_idx_to_tour corresponds to the tour end represented by the i-th network input. 
            nn_input_loc = nn_input[i, :2]
            index_in_tour = el[1]
            cust_id = el[0][index_in_tour][0]
            loc = mdvrp.locations[cust_id]

            self.assertTrue(np.array_equal(nn_input_loc, loc))
            
        # test open_nn_input_idx
        # Open indices are exactly the non-depot ends of incomplete tours
        for idx in mdvrp.open_nn_input_idx:
            self.assertTrue(mdvrp.nn_input_idx_to_tour[idx] is not None)
        self.assertTrue(min(mdvrp.open_nn_input_idx) >= mdvrp.n_depots)

    def test_do_action3(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)

        print(f"\n in test_do_action3:")
        print("mdvrp.solution:")
        for el in mdvrp.solution:
            print(el)
        print("\n")
        print("mdvrp.incomplete_tours:")
        for i, el in enumerate(mdvrp.incomplete_tours):
            print(f"{i} - {el}")
        print("\n")

        # connect incomplete tour #6 [[3, 0, 2], [7, 6, 10]] to depot 3, i.e. [[3, 0, 2]] 
        nn_input_update, cur_nn_input_idx = mdvrp.do_action(
            id_from=10, # incomplete tour #6 is index 10 in input vector
            id_to=2) # depot 3 is index 2 in input vector


        result = [[10, 0, 0]]
        self.assertEqual(result, nn_input_update)

        print(f"DEBUG: nn_input_update after do_action:")
        if nn_input_update is not None:
            for el in nn_input_update:
                print(el)

        print("mdvrp.solution:")
        for el in mdvrp.solution:
            print(el)
        print(f"mdvrp.nn_input_idx_to_tour:")
        for el in mdvrp.nn_input_idx_to_tour:
            print(el)

    def test_do_action4(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)

        print(f"\n in test_do_action3:")
        print("mdvrp.solution:")
        for el in mdvrp.solution:
            print(el)
        print("\n")
        print("mdvrp.incomplete_tours:")
        for i, el in enumerate(mdvrp.incomplete_tours):
            print(f"{i} - {el}")
        print("\n")
        demand = np.asarray(mdvrp.demand)
        demand1 = sum(demand[[c[0] for c in mdvrp.incomplete_tours[2]]])
        demand2 = sum(demand[[c[0] for c in mdvrp.incomplete_tours[4]]])

        print(f"demands of tour: {demand[[c[0] for c in mdvrp.incomplete_tours[2]]]}")
        print(f"demands of tour: {demand[[c[0] for c in mdvrp.incomplete_tours[4]]]}")

        # connect incomplete tour #2 [[2, 0, 1], [20, 3, None], [17, 4, None], [18, 9, 6]] to incomplete tour 4 [[21, 9, 8]]
        nn_input_update, cur_nn_input_idx = mdvrp.do_action(
            id_from=6, # incomplete tour #2 is index 6 in input vector
            id_to=8) # incomplete tour #4 is index 8 in input vector


        r1 = [6, 0, 0]
        r2 = [8, demand1+demand2, 3]
        self.assertTrue(r1 in nn_input_update)
        self.assertTrue(r2 in nn_input_update)

        print(f"DEBUG: nn_input_update after do_action:")
        if nn_input_update is not None:
            for el in nn_input_update:
                print(el)

        print("mdvrp.solution:")
        for el in mdvrp.solution:
            print(el)
        print(f"mdvrp.nn_input_idx_to_tour:")
        for el in mdvrp.nn_input_idx_to_tour:
            print(el)


    def test__rebuild_idx_mapping(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(10)
        mdvrp.destroy_point_based(p=0.3, rng=rng)
        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        
        #print(f"Destroyed solution:")
        #for el in mdvrp.solution:
        #    print(f"\t{el}")
        #print("\n") 
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print(f"nn_input_idx_to_tour:")
        #for el in mdvrp.nn_input_idx_to_tour:
        #    print(f"\t{el}")
        #print("\n") 
        #print("\n") 
        
        result = mdvrp._rebuild_idx_mapping()
        known_result = [
                [[[0, 0, 0]], 0],
                [[[1, 0, 1]], 0],
                [[[2, 0, 2]], 0],
                [[[3, 0, 3]], 0],
                [[[1, 0, 1], [19, 3, 4]], 1],
                [[[16, 4, 5]], 0],
                [[[17, 9, 6]], 0],
                [[[7, 6, 7]], 0],
                [[[20, 9, 8], [5, 5, None], [1, 0, 1]], 0],
                [[[18, 3, 9]], 0],
                [[[13, 3, 10]], 0]]

        self.assertEqual(result, known_result)

    def test__rebuild_idx_mapping2(self):
        mdvrp = self.mdvrp_instance
        mdvrp.create_initial_solution()
        rng = np.random.default_rng(1234)
        mdvrp.destroy_point_based(p=0.3, rng=rng)
        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        
        print(f"Destroyed solution:")
        for el in mdvrp.solution:
            print(f"\t{el}")
        print("\n") 
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"nn_input_idx_to_tour:")
        for el in mdvrp.nn_input_idx_to_tour:
            print(f"\t{el}")
        print("\n") 
        print("\n") 
        
        result = mdvrp._rebuild_idx_mapping()
        known_result = [
                [[[0, 0, 0]], 0],
                [[[1, 0, 1]], 0],
                [[[2, 0, 2]], 0],
                [[[3, 0, 3]], 0],
                [[[19, 3, 4]], 0],
                [[[16, 4, 5]], 0],
                [[[17, 9, 6]], 0],
                [[[7, 6, 7]], 0],
                [[[20, 9, 8], [5, 5, None], [1, 0, 1]], 0],
                [[[3, 0, 3], [18, 3, 9]], 1],
                [[[13, 3, 10]], 0]]

        self.assertEqual(result, known_result)

    def test_get_mask(self):
        rng = np.random.default_rng(1234)
        batch_size = 100 

        config = argparse.Namespace(
                #instance_blueprint='MD_7', 
                instance_blueprint='MD_6', 
                problem_type='mdvrp',
                seed=0,
                capacity=6,
                device='cpu',
                split_delivery=False,
                )

        # generate batch_size instances
        training_set = create_dataset(size=batch_size, config=config,
                                  create_solution=True, use_cost_memory=False, seed=config.seed)
        for instance in training_set:
            instance.destroy_point_based(p=0.3, rng=rng)

        # Create batch input
        input_size      = max([ins.get_max_nb_input_points() for ins in training_set])
        static_input    = np.zeros((batch_size, input_size, 2))
        dynamic_input   = np.zeros((batch_size, input_size, 2), dtype='int')
        nn_origin_indices  = np.zeros((batch_size), dtype=int)
        for i, instance in enumerate(training_set):
            static_nn_input, dynamic_nn_input = instance.get_network_input(input_size) #this updates open_nn_input_idx
            static_input[i]     = static_nn_input
            dynamic_input[i]    = dynamic_nn_input
            if (nn_origin_indices[i] == 0 or int(nn_origin_indices[i]) in instance.depot_indices): #and not instance_repaired[i]:
                if rng is None:
                    nn_origin_indices[i] = np.random.choice(instance.open_nn_input_idx, 1).item()
                else:
                    nn_origin_indices[i] = rng.choice(instance.open_nn_input_idx, 1).item()

        static_input    = torch.from_numpy(static_input).to(config.device).float()
        dynamic_input   = torch.from_numpy(dynamic_input).to(config.device).long()
        mask            = get_mask(nn_origin_indices, dynamic_input, training_set, config, config.capacity).to(config.device)
        # assertions
        for i, instance in enumerate(training_set):
            origin = nn_origin_indices[i]
            self.assertEqual(mask[i, origin], False) # assert origin cannot be sampled again
            origin_tour, origin_pos = instance.nn_input_idx_to_tour[origin]
            same_tour_idxs = [
                j for j, entry in enumerate(instance.nn_input_idx_to_tour)
                if entry is not None and entry[0] is origin_tour
            ]
            
            assert len(same_tour_idxs) <= 2
            for j in same_tour_idxs:
                self.assertFalse(mask[i, j], f"Index {j} points to same tour as origin {origin} but is not masked")

            same_tour_idxs = [
                j for j, entry in enumerate(instance.nn_input_idx_to_tour)
                if entry is not None and entry[0] is origin_tour
            ]
            for j in same_tour_idxs:
                self.assertFalse(mask[i, j], f"Index {j} points to same tour as origin {origin} but is not masked")


            #assert on tour and pos
            self.assertEqual(origin_tour[origin_pos][2], origin)

            home_depot = get_depot(origin_tour, instance.depot_indices)
            if home_depot is not None:
                depot_indices = deepcopy(instance.depot_indices)
                depot_indices = [idx for idx in depot_indices if idx != home_depot]
                known_sol = torch.zeros(len(depot_indices), dtype=torch.bool, device=mask.device)
                # allow connecting only to home_depot among depot nodes
                self.assertTrue(torch.equal(mask[i, depot_indices], known_sol))
                self.assertEqual(mask[i, home_depot], True)
            else:
                self.assertEqual(np.array_equal(mask[i, instance.depot_indices], np.array([True]*len(instance.depot_indices))), True)

            #capacity masking assertions
            customers_of_tour = [cust[0] for cust in origin_tour]
            sum_of_demands = sum([instance.demand[cust] for cust in customers_of_tour])
            self.assertEqual(dynamic_input[i, origin][0], sum_of_demands)

            for idx, el in enumerate(mask[i]):
                if el.item() is True:
                    cand_tour, cand_pos = instance.nn_input_idx_to_tour[idx]
                    cand_customer = cand_tour[cand_pos][0]
                    cand_customer_demand = instance.demand[cand_customer]
                    self.assertTrue(cand_customer_demand + origin_tour[origin_pos][1] <= instance.capacity)


