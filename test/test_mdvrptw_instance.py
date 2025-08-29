import unittest
import numpy as np
from vrp.mdvrptw_problem import MDVRPTWInstance
from search import destroy_instances
SEED = 0

class MDVRPTW_instance_test(unittest.TestCase):
    """ Tests based on the instance generated in setUp() method.
        Hand written numbers, see also `locations.ods` file as a reference """
    def setUp(self):
        self.depot_indices = [0, 1, 2, 3]
        self.locations = np.array(
                    [[0.53133224, 0.28243662],
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
        self.tw_options = {
                'tw_min': 0, 
                'tw_max': 1000, 
                'avg_window': 30,
                'min_window': 10,
                'late_coeff': 10,
                'early_coeff': 10,
                }
        self.capacity = 50
        self.speed = 0.1
        self.service_time = 5
        self.time_windows = [
            [  0, 1000],
            [  0, 1000],
            [  0, 1000],
            [  0, 1000],
            [669,  733],
            [540,  552],
            [ 92,  220],
            [802,  858],
            [719,  729],
            [756,  838],
            [280,  322],
            [311,  414],
            [654,  858],
            [  0,  133],
            [430,  496],
            [506,  542],
            [123,  217],
            [785,  858],
            [682,  858],
            [160,  311],
            [746,  858]]

        # assert on initial data
        self.assertEqual(len(self.demand), len(self.locations))
        self.assertEqual(len(self.time_windows), len(self.locations))

        for i in self.depot_indices:
            self.assertEqual(self.demand[i], 0)
            self.assertEqual(self.time_windows[i], [0, 1000])

        self.instance = MDVRPTWInstance(
                depot_indices       = self.depot_indices,
                locations           = self.locations,
                original_locations  = self.locations,
                demand              = self.demand,
                time_windows        = self.time_windows,
                capacity            = self.capacity,
                speed               = self.speed,
                tw_options          = self.tw_options,
                service_time        = self.service_time,
                )

    
    def test_constructor(self):
        self.assertEqual(self.instance.depot_indices, self.depot_indices)
        #self.assertEqual(self.instance.locations, self.locations)
        self.assertEqual(np.array_equal(self.instance.locations, self.locations), True)
        self.assertEqual(np.array_equal(self.instance.original_locations, self.original_locations), True)
        self.assertEqual(np.array_equal(self.instance.demand, self.demand), True)
        self.assertEqual(np.array_equal(self.instance.capacity, self.capacity), True)

    def test_get_nearest_depot(self):
        customer_to_depot_known = {
                18: 3, 13: 3, 
                9: 2, 10: 2, 11: 2, 14: 2, 6: 2,
                7: 1, 5: 1, 16: 1, 17: 1, 19: 1, 20: 1,
                4: 0, 8: 0, 12: 0, 15: 0}
        for cust in iter(customer_to_depot_known):
            cust = cust
            result = self.instance.get_nearest_depot(cust)
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
            mask = np.array([True] * (self.instance.nb_customers + self.instance.n_depots))
            mask[loc] = False
            mask[self.instance.depot_indices] = False
            result = self.instance.get_n_closest_locations_to(loc, mask, 1)[0] # testing first element only for now
            self.assertEqual(result, known_sol, msg=f"Failed on customer {loc}")


    def test_create_initial_solution(self):
        """ Just checks validy of solution """
        instance = self.instance
        instance.create_initial_solution(tw_options=self.tw_options)
        sol = instance.solution

        # check something was created
        self.assertTrue(instance.solution is not None)
        self.assertTrue(instance.solution_schedule is not None)
        # check first solution tours (should be depots only)
        for i in range(instance.n_depots):
            self.assertEqual(instance.solution[i][0][0], instance.depot_indices[i])
            self.assertEqual(instance.solution[i][0][-1], instance.depot_indices[i])

        #check len of solution and solution_schedule
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

    def test_destroy(self):
        self.instance.create_initial_solution(tw_options=self.tw_options)
        customers_to_remove = [4, 5, 14, 19, 6, 15, 17, 18]
        self.instance.destroy(customers_to_remove)
        sol = self.instance.solution
        incomplete_tours = self.instance.incomplete_tours

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
        self.instance.create_initial_solution(tw_options=self.tw_options)
        rng = np.random.default_rng()
        point = np.array([[0.14365113, 0.5307886 ]])
        p = 0.1
        self.instance.destroy_point_based(p=p, point=point, rng=rng)
        sol = self.instance.solution
        incomplete_tours = self.instance.incomplete_tours
        self.assertIn(member=[[10, 2, None]], container=sol)
        self.assertIn(member=[[10, 2, None]], container=incomplete_tours)

        point = np.array([[0.36975687, 0.91092584]])
        p = 0.1
        self.instance.destroy_point_based(p=p, point=point, rng=rng)
        sol = self.instance.solution
        incomplete_tours = self.instance.incomplete_tours
        self.assertIn(member=[[20, 9, None]], container=sol)
        self.assertIn(member=[[20, 9, None]], container=incomplete_tours)

        point = np.array([[0.79329399, 0.1709594 ]])
        p = 0.1
        self.instance.destroy_point_based(p=p, point=point, rng=rng)
        sol = self.instance.solution
        incomplete_tours = self.instance.incomplete_tours
        self.assertIn(member=[[13, 3, None]], container=sol)
        self.assertIn(member=[[13, 3, None]], container=incomplete_tours)

    def test__get_incomplete_tours(self):
        self.instance.create_initial_solution(tw_options=self.tw_options)
        incomplete_tours = self.instance.incomplete_tours
        self.assertEqual(incomplete_tours, None)

        customers_to_remove = [4, 5, 14, 19, 6, 15, 17, 18]
        self.instance.destroy(customers_to_remove)
        sol = self.instance.solution
        incomplete_tours = self.instance._get_incomplete_tours()

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
        self.instance.create_initial_solution(tw_options=self.tw_options)

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

        self.instance.solution = solution 
        self.instance.incomplete_tours = incomplete_tours 
        n_depots = self.instance.n_depots 
        self.assertEqual(n_depots, 4) #debug
        non_depot_tour_ends = 6 #see comments above
        known_max_nb_input_points = n_depots + non_depot_tour_ends
        max_nb_input_points = self.instance.get_max_nb_input_points()
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

        self.instance.solution = solution 
        self.instance.incomplete_tours = incomplete_tours 
        n_depots = self.instance.n_depots 
        self.assertEqual(n_depots, 4) #debug
        non_depot_tour_ends = 9 #see comments above
        known_max_nb_input_points = n_depots + non_depot_tour_ends
        max_nb_input_points = self.instance.get_max_nb_input_points()
        self.assertEqual(max_nb_input_points, known_max_nb_input_points) 

    def test_get_network_input(self):
        mdvrp = self.instance
        mdvrp.create_initial_solution(tw_options=self.tw_options)
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)
        incomplete_tours = mdvrp.incomplete_tours
        input_dim = mdvrp.get_max_nb_input_points()

        nn_input_static, nn_input_dynamic = mdvrp.get_network_input(input_dim)
        nn_input = np.concat((nn_input_static, nn_input_dynamic), axis=1)
       
        print(f"DEBUG: nn_input_static:")
        for el in nn_input_static[:5]:
            print(el)

        print(f"DEBUG: nn_input_dynamic:")
        for el in nn_input_dynamic[:5]:
            print(el)

        for idx, v in enumerate(nn_input):
            if idx < mdvrp.n_depots: # testing inputs corresponding to depots
                self.assertEqual(v[0], mdvrp.locations[mdvrp.depot_indices[idx]][0]) #depot coordinates
                self.assertEqual(v[1], mdvrp.locations[mdvrp.depot_indices[idx]][1]) #depot coordinates
                self.assertEqual(v[2], mdvrp.time_windows[mdvrp.depot_indices[idx]][0]/mdvrp.tw_options['tw_max']) #time window open
                self.assertEqual(v[3], mdvrp.time_windows[mdvrp.depot_indices[idx]][1]/mdvrp.tw_options['tw_max']) #time window close 
                self.assertEqual(-v[4], mdvrp.capacity) #vehicle capacity
                self.assertEqual(v[5], -1) #depot encoding = -1 
            else:
                tour, i = mdvrp.nn_input_idx_to_tour[idx]
                customer = tour[i][0]
                if len(tour)>1:
                    sum_demands = sum([c[1] for c in tour])
                    self.assertEqual(v[0], mdvrp.locations[customer][0]) #customer coordinates
                    self.assertEqual(v[1], mdvrp.locations[customer][1]) #customer coordinates
                    self.assertEqual(v[2], mdvrp.time_windows[customer][0]/mdvrp.tw_options['tw_max']) #time window open
                    self.assertEqual(v[3], mdvrp.time_windows[customer][1]/mdvrp.tw_options['tw_max']) #time window close 
                    self.assertEqual(v[4], sum_demands) #sum of fulfilled demands
                    if tour[-1][0] in mdvrp.depot_indices:
                        self.assertEqual(v[5], 3) #encoding
                    else:
                        self.assertEqual(v[5], 2) #encoding
                else:
                    demand = tour[0][1]
                    self.assertEqual(v[0], mdvrp.locations[customer][0]) #customer coordinates
                    self.assertEqual(v[1], mdvrp.locations[customer][1]) #customer coordinates
                    self.assertEqual(v[2], mdvrp.time_windows[customer][0]/mdvrp.tw_options['tw_max']) #time window open
                    self.assertEqual(v[3], mdvrp.time_windows[customer][1]/mdvrp.tw_options['tw_max']) #time window close 
                    self.assertEqual(v[4], demand) #sum of fulfilled demands
                    self.assertEqual(v[5], 1) #encoding
                    
    def test_do_action(self):
        mdvrp = self.instance
        mdvrp.create_initial_solution(tw_options=self.tw_options)
        rng = np.random.default_rng(12345)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())
        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #0 [[8, 1, 4]] to incomplete tour #4 [[11, 9, 8]]
        # 
        mdvrp.do_action(
            id_from=4, # incomplete tour #0 is index 4 in input vector
            id_to=8) # incomplete tour #4 is index 8 in input vector
        self.assertIn(member=[[8, 1, 4], [11, 9, 8]], container=mdvrp.solution)

        # connect incomplete tour #1 [[12, 9, 5]] to incomplete tour #0 [[8, 1, 4], [11, 9, 8]]
        # 
        mdvrp.do_action(
            id_from=6, # incomplete tour #1 is index 6 in input vector
            id_to=4) # incomplete tour #0 is index 4 in input vector 
        member = [[12, 9, 5], [8, 1, 4], [11, 9, 8]]
        member_reversed = list(reversed(member)) 
        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )


    def test_do_action2(self):
        mdvrp = self.instance
        mdvrp.create_initial_solution(tw_options=self.tw_options)
        rng = np.random.default_rng(10)
        mdvrp.destroy_point_based(p=0.3, rng=rng)

        nn_input_dynamic, nn_input_static = mdvrp.get_network_input(mdvrp.get_max_nb_input_points())

        #print(f"\nin test_do_action2:\nDEBUG: mdvrp.solution:")
        #for el in mdvrp.solution:
        #    print(f"\t{el}")
        #print("\n")
        #print(f"\nin test_do_action2:\nDEBUG: mdvrp.incomplete_tours:")
        #for el in mdvrp.incomplete_tours:
        #    print(f"\t{el}")
        #print("\n")

        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #2 [[17, 9, 6]] to incomplete tour #0 [[1, 0, 1], [19, 3, 4]]
        # 
        #print(f"\tnn_input:")
        #for el in nn_input:
        #    print(f"\t{el}")
        #print(f"\tmdvrp.nn_input_idx_to_tour")
        #for el in mdvrp.nn_input_idx_to_tour:
        #    print(f"\t{el}")
        #print(f"\tfrom tour: {mdvrp.incomplete_tours[2]}")
        #print(f"\tto tour: {mdvrp.incomplete_tours[0]}")
        mdvrp.do_action(
            id_from=6, # incomplete tour #2 is index 6 in input vector
            id_to=4) # incomplete tour #0 is index 4 in input vector
        self.assertIn(member=[[1, 0, 1], [19, 3, 4], [17, 9, 6]], container=mdvrp.solution)

        #print(f"\nin test_do_action2:\nDEBUG: mdvrp.solution:")
        #for el in mdvrp.solution:
        #    print(f"\t{el}")
        #print("\n")
        #print(f"\nin test_do_action2:\nDEBUG: mdvrp.incomplete_tours:")
        #for el in mdvrp.incomplete_tours:
        #    print(f"\t{el}")
        #print("\n")

        nn_input = np.concat((nn_input_dynamic, nn_input_static), axis=1)
        # connect incomplete tour #2 [[17, 9, 6]] to incomplete tour #0 [[1, 0, 1], [19, 3, 4]]
        # 
        #print(f"\tnn_input:")
        #for el in nn_input:
        #    print(f"\t{el}")
        #print(f"\tmdvrp.nn_input_idx_to_tour")
        #for el in mdvrp.nn_input_idx_to_tour:
        #    print(f"\t{el}")
        #print(f"\tfrom tour: {mdvrp.incomplete_tours[2]}")
        #print(f"\tto tour: {mdvrp.incomplete_tours[0]}")
        # connect incomplete tour #3 [[20, 9, 8], [5, 5, None], [1, 0, 1]] to depot [[1, 0, 1]]
        # 
        mdvrp.do_action(
            id_from=7, # incomplete tour #3 is index 7 in input vector
            id_to=1) # depot 1 is index 1 in input vector 

        member = [[1, 0, 1], [20, 9, 8], [5, 5, None], [1, 0, 1]]
        member_reversed = list(reversed(member)) 
        self.assertTrue(
                member in mdvrp.solution or member_reversed in mdvrp.solution,
                f"Neither: \n{member} \nnor\n {member_reversed} \nfound in solution"
                )


    def test__rebuild_idx_mapping(self):
        mdvrp = self.instance
        mdvrp.create_initial_solution(tw_options=self.tw_options)
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
        mdvrp = self.instance
        mdvrp.create_initial_solution(tw_options=self.tw_options)
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
