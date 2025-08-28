import numpy as np
import torch
# Based on vrp_problem.py

class MDVRPTWInstance():
    """ 
    Generalization of VRPInstance to MD case with Time Windows.

    Attributes 
    ----------
    nb_customers: int
        Number of customers
    n_depots: int
        Number of depots
    depot_indices: list
        List of depot indices
    customer_indices: list
        List of customer indices
    locations: np.ndarray
        Coordinates of all locations in the interval [0, 1]
    original_locations: np.ndarray
        Original coordinates of locations (used to compute objective value)
    demand: np.ndarray 
        Demand for each customer (integer)
    time_windows: np.ndarray
        List of time windows (list of lists with two elements each)
    capacity: int
        Capacity of the vehicles
    speed: float
        Speed of vehicle (unit of distance/unit of time)
    solution: list
        List of tours
    solution_schedule: list
        List of scheduled times for each tour in format [estimated arrival, estimated departure]
    nn_input_idx_to_tour: list
        List of network inputs (see description in __init__)
    open_nn_input_idx: list
        List of indices of inputs that have not been visited
    incomplete_tours: list
        List of incomplete tours of self.solution
    distance_matrix: np.ndarray
        Float array containing distances for each pair of nodes
    service_time: np.array
        Array of service times for each customer
    tw_options: dict
        Dict containing time window config options
    """

    def __init__(
            self, 
            depot_indices, 
            locations, 
            original_locations, 
            demand, 
            time_windows, 
            capacity, 
            speed,
            tw_options, 
            service_time=5,
            distance_matrix=None):
        self.nb_customers = len(locations)-len(depot_indices)
        assert self.nb_customers > 0, f"Error in nb_customers"
        self.depot_indices = depot_indices
        self.n_depots = len(self.depot_indices)
        assert len(self.depot_indices) > 0, f"Insufficient depots"
        self.customer_indices = [i for i in range(self.nb_customers + self.n_depots) if i not in self.depot_indices]
        self.locations = np.array(locations)  # coordinates of all locations in the interval [0, 1]
        self.original_locations = np.array(original_locations)  # original coordinates of locations (used to compute objective
        # value)
        self.time_windows = np.array(time_windows)
        self.demand = np.array(demand)  # demand for each customer (integer). Values are divided by capacity right before being
        # fed to the network
        self.capacity = capacity  # capacity of the vehicle
        self.speed = speed
        assert self.capacity > 1
        assert self.speed > 0

        self.solution = None  # List of tours. Each tour is a list of location elements. Each location element is a
        # list with three values [i_l, d, i_n], with i_l being the index of the location,
        # d being the fulfilled demand of customer i_l by that tour, and
        # i_n being the index of the associated network input.
        self.solution_schedule = None
        self.nn_input_idx_to_tour = None  # After get_network_input() has been called this is a list where the
        # i-th element corresponds to the tour end represented by the i-th network input. If the network
        # points to an input, this allows us to find out, which tour end that input corresponds to.
        self.open_nn_input_idx = None  # List of idx of those nn_inputs that have not been visited
        self.incomplete_tours = None  # List of incomplete tours of self.solution
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        else:
            self.distance_matrix = np.full((self.nb_customers + self.n_depots, self.nb_customers + self.n_depots), np.nan, dtype="float")
        self.fill_distance_matrix(round=False)
        self.service_time = np.full_like(self.demand, fill_value=service_time)
        self.tw_options = tw_options

    def get_n_closest_locations_to(self, origin_location_id, mask, n):
        """Return the idx of the n closest locations (Euclidean) sorted by distance."""
        locs = self.locations
        idxs = np.flatnonzero(mask)                       
        if idxs.size == 0:
            return np.array([], dtype=int)
    
        origin = locs[origin_location_id]
        diffs = locs[idxs] - origin
        dists = np.linalg.norm(diffs, axis=1) 
    
        # Take the n smallest distances among the masked indices
        take = min(n, idxs.size)
        nearest_masked_order = np.argsort(dists)[:take]   # positions within idxs
        return idxs[nearest_masked_order]                 # original indices into self.locations


    def get_nearest_depot(self, loc):
        """ Return nearest depot to loc, based on Euclidian distance. """
        x,y = self.locations[loc]
        min_dist = np.inf 
        closest_depot = None
        for d in self.depot_indices:
            dx, dy = self.locations[d]
            dist = np.sqrt((dx-x)**2 + (dy-y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_depot = d

        return closest_depot 

    def get_n_most_urgent_custs_to(
        self,
        start_idx,
        candidates,
        current_time,
        w=(1.0, 1000.0, 0.5, 1e-3),
        n=1,
        served_mask=None,
        valid_capacity_mask=None,
    ):
        """
        Compute the urgency scores of all candidates for node 'start_idx' and return top n.
        
        Parameters:
            start_idx: int
                index of starting customer
            candidates: np.array
                list of nodes ids to consider
            current_time: float
                Time of departure of start_idx
            w: float array
                weights (see scores formula)
            n: int
                How many customers to return
            served_mask: np.array
                optional boolean mask aligned with candidates. True if customer has already been
                visited/served in the current solution
            valid_capacity_mask: np.array
                optional boolean mask aligned with candidates. True if serving that customer
                would fit the remaining vehicle capacity.

        Returns:
            scores: float array
                array of urgency scores for candidates 
            departs: float array
                array of departure times from candidates
        """
        candidates = np.asarray(candidates, dtype=int)
    
        tw     = self.time_windows[candidates]               
        s_time = self.service_time[candidates]                
        tt     = self.distance_matrix[start_idx, candidates]/self.speed 
    
        arrival       = current_time + tt
        start_service = np.maximum(arrival, tw[:, 0])
        early         = np.maximum(0.0, tw[:, 0] - arrival)
        late          = np.maximum(0.0, start_service - tw[:, 1])
        depart        = start_service + s_time
    
        scores = (w[0] * tt +
                  w[1] * late +
                  w[2] * early +
                  w[3] * tw[:, 1])
    
        if served_mask is not None:
            served_mask = np.asarray(served_mask, dtype=bool)
            scores = np.where(served_mask, np.inf, scores)
        if valid_capacity_mask is not None:
            valid_capacity_mask = np.asarray(valid_capacity_mask, dtype=bool)
            scores = np.where(~valid_capacity_mask, np.inf, scores)
    
        if not np.isfinite(scores).any():
            return (None, None, None, None)
    
        n = int(min(n, len(candidates)))
        top_idx = np.argpartition(scores, n - 1)[:n]
        top_idx = top_idx[np.argsort(scores[top_idx])]  
    
        best_ids       = np.round(candidates[top_idx])
        best_scores    = np.round(scores[top_idx])
        best_starts    = np.round(start_service[top_idx])
        best_departs   = np.round(depart[top_idx])
    
        return (best_ids, best_scores, best_starts, best_departs)


    def create_initial_solution(self, config, tw_options):
        """Create an initial solution for this instance using a greedy heuristic."""
        tw_min = tw_options['tw_min']
        tw_max = tw_options['tw_max']

        #1 create clusters for each depot
        depot_to_customer = {} # dict mapping depot id to list of customers in cluster
        for depot in self.depot_indices:
            depot_to_customer[depot] = []
        
        for idx, _ in enumerate(self.locations):
            if idx in self.depot_indices:
                continue 
            nearest_depot = self.get_nearest_depot(idx)
            depot_to_customer[nearest_depot].append(idx)

        self.solution_schedule = []
        self.solution = []
        for input_idx, depot in enumerate(self.depot_indices):
            self.solution.append([[depot, 0, input_idx]]) # for depot only
            self.solution_schedule.append([[tw_min, tw_min]])

        # 2 do normal NLNS VRP greedy solution inside each cluster
        # use mask to only see some customers
        for input_idx, depot in enumerate(self.depot_indices):
            available_customers = depot_to_customer[depot]
            mask = np.array([False] * (self.nb_customers + self.n_depots))
            # enable only current depot and current available customers
            mask[depot] = False 
            mask[available_customers] = True
            cur_load = self.capacity
            self.solution.append([[depot, 0, input_idx]]) # to start route
            self.solution_schedule.append([[tw_min, tw_min]]) 

            best = None
            best_score = float('inf')
            route = []

            while mask.any() > 0:
                current_time = self.solution_schedule[-1][-1][-1]
                if current_time == tw_max: # last route was closed
                    current_time = tw_min
                most_urgent_cust_idx, _, most_urgent_cust_start_time, most_urgent_cust_dep_time = self.get_n_most_urgent_custs_to(
                        start_idx = self.solution[-1][-1][0],
                        candidates = np.asarray(available_customers, dtype=int),
                        current_time = current_time,
                        n=1)
                    # if demand and time are valid then append to last modified route
                cand_demand = self.demand[most_urgent_cust_idx]
                time_to_depot = np.round(self.distance_matrix[most_urgent_cust_idx, depot]/self.speed).item()
                if (cand_demand <= cur_load) and (most_urgent_cust_dep_time + time_to_depot <= tw_max):
                    mask[most_urgent_cust_idx] = False
                    available_customers.remove(most_urgent_cust_idx)
                    self.solution[-1].append([int(most_urgent_cust_idx.item()), int(self.demand[most_urgent_cust_idx].item()), None])
                    new_schedule = [most_urgent_cust_start_time.item(), most_urgent_cust_dep_time.item()]
                    self.solution_schedule[-1].append(new_schedule)
                    cur_load -= self.demand[most_urgent_cust_idx]
                else: #otherwise start new route and close previous
                    i = self.solution[-1][-1][0]
                    self.solution[-1].append([depot, 0, input_idx])
                    time_to_depot = np.round(self.distance_matrix[i, depot]/self.speed).item()
                    self.solution_schedule[-1].append([current_time + time_to_depot, tw_max])
                    self.solution.append([[depot, 0, input_idx]])
                    self.solution_schedule.append([[tw_min, tw_min]])
                    cur_load = self.capacity

            last_cust = self.solution[-1][-1][0]
            time_to_depot = np.round(self.distance_matrix[last_cust, depot]/self.speed).item()
            current_time = self.solution_schedule[-1][-1][-1]
            self.solution[-1].append([depot, 0, input_idx])
            self.solution_schedule[-1].append([current_time + time_to_depot, tw_max])

    def fill_distance_matrix(self, round):
        side = self.n_depots+self.nb_customers
        cc = np.zeros(shape=(side, side))
        for i in range(side):
            for j in range(side):
                cc[i, j] = np.sqrt((self.original_locations[i, 0] - self.original_locations[j, 0]) ** 2
                + (self.original_locations[i, 1] - self.original_locations[j, 1]) ** 2)
        if round:
            cc = np.round(cc)
        self.distance_matrix = cc

    def get_sum_early_late_mins(self, round):
        """ Returns the sum of early minutes and the sum of late minutes of complete/incomplete solution """


    def get_costs_memory(self, round):
        """ Return the cost of the current complete solution. Uses a memory to improve performance.
            Cost is given by distance between nodes + late_coeff*(sum of late mins) + early_coeff*(sum early mins)"""
        c = 0
        a = self.tw_options['early_coeff']
        b = self.tw_options['late_coeff']
        #for t in self.solution:
        for tour, schedule in zip(self.solution, self.solution_schedule):
            #check that first position element of tour tour starts at a depot
            if tour[0][0] not in self.depot_indices or tour[-1][0] not in self.depot_indices:
                raise Exception("Incomplete solution.")
            for i in range(0, len(tour) - 1):
                from_idx = tour[i][0] 
                to_idx = tour[i + 1][0]
                #distance part
                if np.isnan(self.distance_matrix[from_idx, to_idx]):
                    cc = np.sqrt((self.original_locations[from_idx, 0] - self.original_locations[to_idx, 0]) ** 2
                                 + (self.original_locations[from_idx, 1] - self.original_locations[to_idx, 1]) ** 2)
                    if round:
                        cc = np.round(cc)
                    self.distance_matrix[from_idx, to_idx] = cc
                    c += cc
                else:
                    c += self.distance_matrix[from_idx, to_idx]
                #schedule departure part
                planned_arrival, planned_departure = schedule[i] 
                delta_early = max(0, self.time_windows[to_idx, 0] - planned_arrival)
                
                delta_late = max(0, planned_departure - self.time_windows[to_idx, 1])
                cc = a*delta_early + b*delta_late
                if round:
                    cc = np.round(cc)
                c += cc 
        return c

    def get_costs(self, round):
        """Return the cost of the current complete solution."""
        c = 0
        for t, schedule in zip(self.solution, self.solution_schedule):
            if t[0][0] not in self.depot_indices or t[-1][0] not in self.depot_indices:
                raise Exception("Incomplete solution.")
            for i in range(0, len(t) - 1):
                from_idx = t[i][0] 
                to_idx = t[i + 1][0]
                cc = np.sqrt((self.original_locations[from_idx, 0] - self.original_locations[to_idx, 0]) ** 2
                             + (self.original_locations[from_idx, 1] - self.original_locations[to_idx, 1]) ** 2)

                #schedule departure part
                planned_arrival, planned_departure = schedule[i] 
                delta_early = max(0, self.time_windows[to_idx, 0] - planned_arrival)
                
                delta_late = max(0, planned_departure - self.time_windows[to_idx, 1])
                cc += a*delta_early + b*delta_late
                if round:
                    cc = np.round(cc)
                c += cc
        return c

    def get_costs_incomplete(self, round):
        """Return the cost of the current incomplete solution."""
        c = 0
        for tour in self.solution:
            if len(tour) <= 1:
                continue
            for i in range(0, len(tour) - 1):
                cc = np.sqrt((self.original_locations[tour[i][0], 0] - self.original_locations[tour[i + 1][0], 0]) ** 2
                             + (self.original_locations[tour[i][0], 1] - self.original_locations[
                    tour[i + 1][0], 1]) ** 2)
                if round:
                    cc = np.round(cc)
                c += cc
        return c

    def destroy(self, customers_to_remove_idx):
        """Remove the customers with the given idx from their tours. This creates an incomplete solution."""
        self.incomplete_tours = []
        st = []  # solution tours

        removed_customer_idx = []

        for tour in self.solution:
            last_split_idx = 0
            for i in range(1, len(tour) - 1):
                if tour[i][0] in customers_to_remove_idx:
                    # Create two new tours:
                    # The first consisting of the tour from the depot or from the last removed customer to the
                    # customer that should be removed
                    if i > last_split_idx and i > 1:
                        new_tour_pre = tour[last_split_idx:i]
                        st.append(new_tour_pre)
                        self.incomplete_tours.append(new_tour_pre)

                    # The second consisting of only the customer to be removed
                    customer_idx = tour[i][0]
                    if customer_idx not in removed_customer_idx:  # make sure the customer has not already been
                        # extracted from a different tour
                        demand = int(self.demand[customer_idx])
                        new_tour = [[customer_idx, demand, None]]
                        st.append(new_tour)
                        self.incomplete_tours.append(new_tour)
                        removed_customer_idx.append(customer_idx)
                    last_split_idx = i + 1

            if last_split_idx > 0:
                # Create another new tour consisting of the remaining part of the original tour
                if last_split_idx < len(tour) - 1:
                    new_tour_post = tour[last_split_idx:]
                    st.append(new_tour_post)
                    self.incomplete_tours.append(new_tour_post)
            else:  # add unchanged tour
                st.append(tour)

        self.solution = st

    def destroy_random(self, p, rng):
        """Random destroy. Select customers that should be removed at random and remove them from tours."""
        customers_to_remove_idx = rng.choice(range(1, self.nb_customers + 1), int(self.nb_customers * p), replace=True)
                #a=self.customer_indices,
                #size=int(self.nb_customers * p), # degree of destruction
                #replace=False)
        self.destroy(customers_to_remove_idx)

    def destroy_point_based(self, p, rng, point=None):
        """Point based destroy. Select customers that should be removed based on their distance to a random point
         and remove them from tours."""
        nb_customers_to_remove = int(self.nb_customers * p)
        if point is None:
            random_point = rng.random((1,2))
        else:
            random_point = point
        customer_locations = self.locations[self.customer_indices]
        #dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1) 
        dist = np.sum((customer_locations - random_point) ** 2, axis=1) #squared euclidian distance
        closest_customers_idx = np.argsort(dist)[:nb_customers_to_remove] + self.n_depots 
        self.destroy(closest_customers_idx)

    def destroy_tour_based(self, p, rng):
        """Tour based destroy. Remove all tours closest to a randomly selected point from a solution."""
        # Make a dictionary that maps customers to tours
        customer_to_tour = {}
        for i, tour in enumerate(self.solution[1:]):
            for e in tour[1:-1]:
                if e[0] in customer_to_tour:
                    customer_to_tour[e[0]].append(i + 1)
                else:
                    customer_to_tour[e[0]] = [i + 1]

        nb_customers_to_remove = int(self.nb_customers * p)  # Number of customer that should be removed
        nb_removed_customers = 0
        tours_to_remove_idx = []
        #random_point = np.random.rand(1, 2)  # Randomly selected point
        random_point = rng.random((1,2))
        dist = np.sum((self.locations[1:] - random_point) ** 2, axis=1)
        closest_customers_idx = np.argsort(dist) + 1

        # Iterate over customers starting with the customer closest to the random point.
        for customer_idx in closest_customers_idx:
            # Iterate over the tours of the customer
            for i in customer_to_tour[customer_idx]:
                # and if the tour is not yet marked for removal
                if i not in tours_to_remove_idx:
                    # mark it for removal
                    tours_to_remove_idx.append(i)
                    nb_removed_customers += len(self.solution[i])

            # Stop once enough tours are marked for removal
            if nb_removed_customers >= nb_customers_to_remove and len(tours_to_remove_idx) > 1:
                break

        # Create the new tours that all consist of only a single customer
        new_tours = []
        removed_customer_idx = []
        for i in tours_to_remove_idx:
            tour = self.solution[i]
            for e in tour[1:-1]:
                if e[0] in removed_customer_idx:
                    for new_tour in new_tours:
                        if new_tour[0][0] == e[0]:
                            new_tour[0][1] += e[1]
                            break
                else:
                    new_tours.append([e])
                    removed_customer_idx.append(e[0])

        # Remove the tours that are marked for removal from the solution
        for index in sorted(tours_to_remove_idx, reverse=True):
            del self.solution[index]

        self.solution.extend(new_tours)  # Add new tours to solution
        self.incomplete_tours = new_tours

    def _get_incomplete_tours(self):
        incomplete_tours = []
        for tour in self.solution:
            if tour[0][0]  not in self.depot_indices or tour[-1][0] not in self.depot_indices:
                incomplete_tours.append(tour)
        return incomplete_tours

    def get_max_nb_input_points(self):
        """For the current instance, returns the number of input vectors required to describe the
        incomplete solution. """
        if self.incomplete_tours is None:
            self.incomplete_tours = self._get_incomplete_tours()

        incomplete_tours = self.incomplete_tours
        nb = self.n_depots  # input point for each depot
        for tour in incomplete_tours:
            if len(tour) == 1:
                nb += 1
            else:
                if tour[0][0] not in self.depot_indices:
                    nb += 1
                if tour[-1][0] not in self.depot_indices:
                    nb += 1
        return nb

    def get_network_input(self, input_size):
        """Generate the tensor representation of an incomplete solution (i.e, a representation of the repair problem).
         The input size must be provided so that the representations of all inputs of the batch have the same size.

        [:, 0] x-coordinates for all points
        [:, 1] y-coordinates for all points
        [:, 2] demand values for all points
        [:, 3] state values for all points

        """
        nn_input = np.zeros((input_size, 4))
        nn_input[:self.n_depots, :2] = self.locations[self.depot_indices]
        nn_input[:self.n_depots, 2] = -1 * self.capacity  # Depots demand
        nn_input[:self.n_depots, 3] = -1  # Depots state

        #nn_input[0, :2] = self.locations[0]  # Depot location
        #nn_input[0, 2] = -1 * self.capacity  # Depot demand
        #nn_input[0, 3] = -1  # Depot state
        network_input_idx_to_tour = [None] * input_size
        for d in range(self.n_depots): 
            network_input_idx_to_tour[d] = [self.solution[d], 0] #IMPORTANT: first part of solution have to be depots!!
        i = self.n_depots 
        destroyed_location_idx = []

        if self.incomplete_tours is None:
            self.incomplete_tours = self._get_incomplete_tours()
        incomplete_tours = self.incomplete_tours
        for tour in incomplete_tours:
            # Create an input for a tour consisting of a single customer
            if len(tour) == 1:
                nn_input[i, :2] = self.locations[tour[0][0]] #coordinates of customer
                nn_input[i, 2] = tour[0][1] # demand of customer
                nn_input[i, 3] = 1 #encoding of single customer route
                tour[0][2] = i # save network input index information in incomplete_tours
                network_input_idx_to_tour[i] = [tour, 0]
                destroyed_location_idx.append(tour[0][0])
                i += 1
            else:
                # Create an input for the first location in an incomplete tour if the location is not the depot
                if tour[0][0] not in self.depot_indices:
                    nn_input[i, :2] = self.locations[tour[0][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, 0]
                    if tour[-1][0] in self.depot_indices: # if route contains (i.e. ends at) a depot
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    tour[0][2] = i # save network input index information in incomplete_tours
                    destroyed_location_idx.append(tour[0][0])
                    i += 1
                # Create an input for the last location in an incomplete tour if the location is not the depot
                if tour[-1][0] not in self.depot_indices:
                    nn_input[i, :2] = self.locations[tour[-1][0]]
                    nn_input[i, 2] = sum(l[1] for l in tour)
                    network_input_idx_to_tour[i] = [tour, len(tour) - 1]
                    tour[-1][2] = i
                    if tour[0][0] in self.depot_indices:
                        nn_input[i, 3] = 3
                    else:
                        nn_input[i, 3] = 2
                    destroyed_location_idx.append(tour[-1][0])
                    i += 1

       # self.open_nn_input_idx = []
       # for tour in incomplete_tours:
       #     if len(tour)>1:
       #        # if first end not depot
       #         if tour[0][0] not in self.depot_indices:
       #             self.open_nn_input_idx.append(tour[0][2])
       #         if tour[-1][0] not in self.depot_indices:
       #             self.open_nn_input_idx.append(tour[-1][2])
       #     if len(tour) == 1:
       #         if tour[0][0] not in self.depot_indices:
       #             self.open_nn_input_idx.append(tour[0][2])

        self.open_nn_input_idx = list(range(self.n_depots, i))
        self.nn_input_idx_to_tour = network_input_idx_to_tour
        return nn_input[:, :2], nn_input[:, 2:]

    def _get_network_input_update_for_tour(self, tour, new_demand):
        """Returns an nn_input update for the tour tour. The demand of the tour is updated to new_demand"""

        nn_input_idx_start = tour[0][2]  # Idx of the nn_input for the first location in tour
        nn_input_idx_end = tour[-1][2]  # Idx of the nn_input for the last location in tour

        # If the tour stars and ends at the depot, no update is required
        if tour[-1][0] in self.depot_indices and tour[0][0] in self.depot_indices:
            #assert tour[-1][0] == tour[0][0] #added by piero: check that tour starts and end at same depot
            return []

        nn_input_update = []
        # Tour with a single location
        if len(tour) == 1:
            if tour[0][0] not in self.depot_indices:
                nn_input_update.append([nn_input_idx_end, new_demand, 1])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, 0]
        else:
            # Tour contains the depot
            if tour[0][0] in self.depot_indices or tour[-1][0] in self.depot_indices:
                # First location in the tour is not the depot
                if tour[0][0] not in self.depot_indices:
                    nn_input_update.append([nn_input_idx_start, new_demand, 3])
                    # update first location
                    self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                # Last location in the tour is not the depot
                elif tour[-1][0] not in self.depot_indices:
                    nn_input_update.append([nn_input_idx_end, new_demand, 3])
                    # update last location
                    self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
            # Tour does not contain the depot
            else:
                # update first and last location of the tour
                nn_input_update.append([nn_input_idx_start, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_start] = [tour, 0]
                nn_input_update.append([nn_input_idx_end, new_demand, 2])
                self.nn_input_idx_to_tour[nn_input_idx_end] = [tour, len(tour) - 1]
        return nn_input_update

    def do_action(self, id_from, id_to):
        """Performs an action. The tour end represented by input with the id id_from is connected to the tour end
         presented by the input with id id_to."""

        tour_from = self.nn_input_idx_to_tour[id_from][0]  # Tour that should be connected
        tour_to = self.nn_input_idx_to_tour[id_to][0]  # to this tour.
        pos_from = self.nn_input_idx_to_tour[id_from][1]  # Position of the location that should be connected in tour_from
        pos_to = self.nn_input_idx_to_tour[id_to][1]  # Position of the location that should be connected in tour_to
        # Exchange tour_from with tour_to or invert order of the tours. This reduces the number of cases that need
        # to be considered in the following.
        if len(tour_from) > 1 and len(tour_to) > 1:
            if pos_from > 0 and pos_to > 0:
                tour_to.reverse()
            elif pos_from == 0 and pos_to == 0:
                tour_from.reverse()
            elif pos_from == 0 and pos_to > 0:
                tour_from, tour_to = tour_to, tour_from
        elif len(tour_to) > 1:
            if pos_to == 0:
                tour_to.reverse()
            tour_from, tour_to = tour_to, tour_from
        elif len(tour_from) > 1 and pos_from == 0:
            tour_from.reverse()

        # Now we only need to consider two cases 1) Connecting an incomplete tour with more than one location
        # to an incomplete tour with more than one location 2) Connecting an incomplete tour (single
        # or multiple locations) to incomplete tour consisting of a single location

        nn_input_update = []  # Instead of recalculating the tensor representation, we only compute an update description.
        # This improves performance.

        # Case 1
        if len(tour_from) > 1 and len(tour_to) > 1:
            combined_demand = sum(l[1] for l in tour_from) + sum(l[1] for l in tour_to)
            assert combined_demand <= self.capacity  # This is ensured by the masking schema

            # The two incomplete tours are combined to one (in)complete tour. All network inputs associated with the
            # two connected tour ends are set to 0 (deactivated)
            nn_input_update.append([tour_from[-1][2], 0, 0])
            nn_input_update.append([tour_to[0][2], 0, 0])
            tour_from.extend(tour_to)

            for t in self.solution:
                if t is tour_to:
                    self.solution.remove(t)
                    break
            else:
                raise ValueError(f"{tour_to} not found in self.solution")
            #self.solution.remove(tour_to)
            nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))

        # Case 2
        if len(tour_to) == 1:
            demand_from = sum(l[1] for l in tour_from)
            combined_demand = demand_from + sum(l[1] for l in tour_to)
            unfulfilled_demand = combined_demand - self.capacity

            # The new tour has a total demand that is smaller than or equal to the vehicle capacity
            if unfulfilled_demand <= 0:
                if len(tour_from) > 1:
                    nn_input_update.append([tour_from[-1][2], 0, 0])
                # Update solution
                tour_from.extend(tour_to)

                for t in self.solution:
                    if t is tour_to:
                        self.solution.remove(t)
                        break
                else:
                    print(f"ERROR: self.solution:")
                    for el in self.solution:
                        print(el)
                    raise ValueError(f"{tour_to} not found in self.solution")
                #self.solution.remove(tour_to)
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_tour(tour_from, combined_demand))
            # The new tour has a total demand that is larger than the vehicle capacity
            else:
                raise NotImplementedError #think about this (split delivery?)
                nn_input_update.append([tour_from[-1][2], 0, 0])
                if len(tour_from) > 1 and tour_from[0][0] not in self.depot_indices:
                    nn_input_update.append([tour_from[0][2], 0, 0])

                # Update solution
                tour_from.append([tour_to[0][0], tour_to[0][1], tour_to[0][2]])  # deepcopy of tour_to
                tour_from[-1][1] = self.capacity - demand_from
                tour_from.append([0, 0, 0])
                if tour_from[0][0] not in self.depot_indices:
                    tour_from.insert(0, [0, 0, 0])
                tour_to[0][1] = unfulfilled_demand  # Update demand of tour_to
                # Generate input update
                nn_input_update.extend(self._get_network_input_update_for_tour(tour_to, unfulfilled_demand))

        # Add depot tours to the solution tours if they were removed
        for idx, d in enumerate(self.depot_indices):
            if self.solution[idx] != [[d, 0, idx]]:
                self.solution.insert(idx, [[d, 0, idx]])
                self.nn_input_idx_to_tour[idx] = [self.solution[idx], 0]

        for update in nn_input_update:
            #if update[2] == 0 and update[0] != 0:
            if update[2] == 0 and update[0] not in self.depot_indices:
                self.open_nn_input_idx.remove(update[0])

        return nn_input_update, tour_from[-1][2]

    def _rebuild_idx_mapping(self):
        len_input = self.get_max_nb_input_points()

        result = [None]*len_input
        for idx, d in enumerate(self.depot_indices):
            result[idx] = [self.solution[idx], 0]
        i = self.n_depots

        for tour in self.incomplete_tours:
            if len(tour) == 1:
                result[i] = [tour, 0]
                i+=1
            else:
                if tour[0][0] not in self.depot_indices:
                    result[i] = [tour, 0]
                    i+=1
                if tour[-1][0] not in self.depot_indices:
                    result[i] = [tour, len(tour) -1]
                    i+=1

        return result


    def verify_solution(self, config):
        """Verify that a feasible solution has been found."""
        d = np.zeros((self.nb_customers + self.n_depots), dtype=int)
        for i in range(len(self.solution)):
            for ii in range(len(self.solution[i])):
                d[self.solution[i][ii][0]] += self.solution[i][ii][1]
        if (self.demand != d).any():
            raise Exception('Solution could not be verified.')

        for tour in self.solution:
            if sum([t[1] for t in tour]) > self.capacity:
                raise Exception('Solution could not be verified.')

        if not config.split_delivery:
            customers = []
            for tour in self.solution:
                for c in tour:
                    if c[0] not in self.depot_indices:
                        customers.append(c[0])
            if len(customers) > len(set(customers)):
                raise Exception('Solution could not be verified.')

    def get_solution_copy(self):
        """ Returns a copy of self.solution"""
        solution_copy = []
        for tour in self.solution:
            solution_copy.append([x[:] for x in tour]) # Fastest way to make a deep copy
        return solution_copy

    def __deepcopy__(self, memo):
        solution_copy = self.get_solution_copy()
        new_instance = MDVRPInstance(
                depot_indices       = self.depot_indices,
                locations           = self.locations, 
                original_locations  = self.original_locations,
                demand              = self.demand,
                capacity            = self.capacity)
        new_instance.solution = solution_copy
        new_instance.distance_matrix = self.distance_matrix

        if self.incomplete_tours is not None:
            new_instance.incomplete_tours = [ [x[:] for x in tour] for tour in self.incomplete_tours ]
        else:
            new_instance.incomplete_tours = None

        return new_instance


def get_mask(origin_nn_input_idx, dynamic_input, instances, config, capacity):
    """ Returns a mask for the current nn_input"""
    device = dynamic_input.device
    batch_size, N, _ = dynamic_input.shape

    # Make sure origin indices are a 1-D torch.LongTensor on the right device
    origin_idx = torch.as_tensor(origin_nn_input_idx, device=device, dtype=torch.long).view(-1)
    assert origin_idx.size(0) == batch_size, f"origin_idx batch {origin_idx.size(0)} != dynamic_input batch {batch_size}"

    # Start with all 'alive' input positions (i.e. all non-zero ones)
    #mask = (dynamic_input[:, :, 1] != 0).cpu().long().numpy()
    mask = (dynamic_input[:, :, 1] != 0).clone()


    # FIRST PART: avoid connecting both ends of the same tour or connecting to itself (creating cycles)
    for i in range(batch_size):
        inst = instances[i]
        depot_indices = inst.depot_indices

        idx_from = origin_nn_input_idx[i]   # for the i-th instance in the batch, this is the index of the tour end
                                            # we want to connect from

        # Find the start of the tour in the nn input
        origin_tour, origin_pos = inst.nn_input_idx_to_tour[idx_from]

        # forbid connecting origin to itself & to the opposite end of the same tour
        mask[i, idx_from] = False
        if origin_pos == 0: #if origin is the start then fetch the end
            idx_same_tour = origin_tour[-1][2]
        else: # if origin is the end then fetch the start
            idx_same_tour = origin_tour[0][2]
        mask[i, idx_same_tour] = False 

        # Same depot rules:
        # determine if origin tour already contains a depot
        home_depot = get_depot(origin_tour, depot_indices)
        if home_depot is not None:
            # allow connecting only to home_depot among depot nodes
            mask[i, depot_indices] = False
            mask[i, depot_indices] = True

            #forbid connecting to another incomplete tour that already contains a different depot
            # here each candidate j represents a tour end
            for j in range(N):
                if not mask[i, j]:
                    continue
                cand_tour, _ = inst.nn_input_idx_to_tour[j]
                if cand_tour is None: #padding beyond current nn inputs
                    mask[i, j] = False
                    continue
                cand_depot = get_depot(cand_tour, depot_indices)
                if cand_depot is not None:
                    if cand_depot != home_depot:
                        mask[i, j] = False
        else:
            # the origin tour has not yet touched any depot:
            # allow all depots for now
            mask[i, depot_indices] = True

    # capacity constraints
    origin_tour_demands = dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 0]
    combined_demand = origin_tour_demands.unsqueeze(1).expand(batch_size, dynamic_input.shape[1]) + dynamic_input[:, :,
                                                                                                    0]

    if config.split_delivery:
        multiple_customer_tour = (dynamic_input[torch.arange(batch_size), origin_nn_input_idx, 1] > 1).unsqueeze(1).expand(
            batch_size, dynamic_input.shape[1])

        # If the origin tour consists of multiple customers mask all tours with multiple customers where
        # the combined demand is > 1
        #mask[multiple_customer_tour & (combined_demand > capacity) & (dynamic_input[:, :, 1] > 1)] = 0
        mask &= ~(multiple_customer_tour & (combined_demand > capacity) & (dynamic_input[:, :, 1] > 1))

        # If the origin tour consists of a single customer mask all tours with demand is >= 1
        #mask[(~multiple_customer_tour) & (dynamic_input[:, :, 0] >= capacity)] = 0
        mask &= ~(~multiple_customer_tour & (dynamic_input[:, :, 0] >= capacity))

    else:
        #mask[combined_demand > capacity] = 0
        mask &= ~(combined_demand > capacity)

    return mask

def get_depot(tour: list, depot_indices: list):
    has_depot = (tour[0][0] in depot_indices) or (tour[-1][0] in depot_indices)
    if has_depot:
        return tour[0][0] if tour[0][0] in depot_indices else tour[-1][0]
    else:
        return None
