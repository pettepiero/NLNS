from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['0'] = InstanceBlueprint(nb_customers=20, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=2,
    demand_type='inter', demand_min=1, demand_max=5, capacity=50, grid_size=1000)  
dataset['1'] = InstanceBlueprint(nb_customers=50, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=5, capacity=50, grid_size=1000)  

dataset['2'] = InstanceBlueprint(nb_customers=100, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=5, capacity=50, grid_size=1000)  

dataset['3'] = InstanceBlueprint(nb_customers=150, n_depots=3, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=50, grid_size=1000)  

dataset['4'] = InstanceBlueprint(nb_customers=300, n_depots=5, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=50, grid_size=1000)  

dataset['5'] = InstanceBlueprint(nb_customers=100, n_depots=3, depot_position='R', customer_position='RC', nb_customer_cluster=7,
    demand_type='inter', demand_min=1, demand_max=10, capacity=20, grid_size=1000)  

dataset['6'] = InstanceBlueprint(nb_customers=100, n_depots=3, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=5, capacity=10, grid_size=1)

dataset['7'] = InstanceBlueprint(nb_customers=20, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=5, capacity=6, grid_size=1000)


dataset['8'] = InstanceBlueprint(nb_customers=100, n_depots=4, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=9, capacity=50, grid_size=1000)

dataset['9'] = InstanceBlueprint(nb_customers=128, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=9, capacity=39, grid_size=1000)
dataset['10'] = InstanceBlueprint(nb_customers=128, n_depots=2, depot_position='E', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=9, capacity=39, grid_size=1000)
dataset['11'] = InstanceBlueprint(nb_customers=20, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=7, demand_type='inter', demand_min=1, demand_max=3, capacity=10, grid_size=1000)
dataset['12'] = InstanceBlueprint(nb_customers=10, n_depots=1, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=1, capacity=4, grid_size=1000)
dataset['13'] = InstanceBlueprint(nb_customers=10, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=1, capacity=4, grid_size=1000)
dataset['14'] = InstanceBlueprint(nb_customers=50, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=5, capacity=20, grid_size=1000)
dataset['15'] = InstanceBlueprint(nb_customers=20, n_depots=2, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=5, capacity=10, grid_size=1000)



dataset['40'] = InstanceBlueprint(nb_customers=10, n_depots=2, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000)  
dataset['240'] = InstanceBlueprint(nb_customers=15, n_depots=2, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=25, grid_size=1000)  
dataset['50'] = InstanceBlueprint(nb_customers=15, n_depots=2, depot_position='R', customer_position='R', nb_customer_cluster=1,
    demand_type='inter', demand_min=1, demand_max=10, capacity=40, grid_size=1000)  

dataset['20'] = InstanceBlueprint(nb_customers=20, n_depots=1, depot_position='R', customer_position='RC', nb_customer_cluster=2, demand_type='inter', demand_min=1, demand_max=1, capacity=4, grid_size=1000)
