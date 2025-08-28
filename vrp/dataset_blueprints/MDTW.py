from vrp.data_utils import InstanceBlueprint

tw_options = {
        'tw_min': 0,
        'tw_max': 1000,
        'avg_window': 100,
        'min_window': 10,
        'late_coeff': 10,
        'early_coeff': 10,
    }
capacity = 50
speed = 10


dataset = {}
dataset['0'] = InstanceBlueprint(problem_type='mdvrptw', nb_customers=20, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=5, capacity=capacity, speed=speed, grid_size=1000, tw_options=tw_options)  

dataset['1'] = InstanceBlueprint(problem_type='mdvrptw', nb_customers=50, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=5, capacity=capacity, speed=speed, grid_size=1000, tw_options=tw_options)

dataset['2'] = InstanceBlueprint(problem_type='mdvrptw', nb_customers=100, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=5, capacity=capacity, speed=speed, grid_size=1000, tw_options=tw_options)  

dataset['3'] = InstanceBlueprint(problem_type='mdvrptw', nb_customers=150, n_depots=3, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=10, capacity=capacity, speed=speed, grid_size=1000, tw_options=tw_options)  

dataset['4'] = InstanceBlueprint(problem_type='mdvrptw', nb_customers=300, n_depots=5, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=10, capacity=capacity, speed=speed, grid_size=1000, tw_options=tw_options)  
