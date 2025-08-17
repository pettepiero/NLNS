from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['1'] = InstanceBlueprint(nb_customers=50, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=5, capacity=50, grid_size=1000)  

dataset['2'] = InstanceBlueprint(nb_customers=100, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=5, capacity=50, grid_size=1000)  

dataset['3'] = InstanceBlueprint(nb_customers=150, n_depots=3, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=50, grid_size=1000)  

dataset['4'] = InstanceBlueprint(nb_customers=300, n_depots=5, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=50, grid_size=1000)  
