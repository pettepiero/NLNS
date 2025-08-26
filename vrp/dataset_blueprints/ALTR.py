from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['10'] = InstanceBlueprint('vrp', 10, 'R', 'R', '0', 'inter', 1, 9, 20, 1, n_depots=1)
dataset['20'] = InstanceBlueprint('vrp', 20, 'R', 'R', '0', 'inter', 1, 9, 30, 1, n_depots=1)
dataset['50'] = InstanceBlueprint('vrp', 50, 'R', 'R', '0', 'inter', 1, 9, 40, 1, n_depots=1)
dataset['100'] = InstanceBlueprint('vrp', 100, 'R', 'R', '0', 'inter', 1, 9, 50, 1, n_depots=1)
dataset['250'] = InstanceBlueprint('vrp', 250, 'R', 'R', '0', 'inter', 1, 9, 60, 1, n_depots=1)
dataset['500'] = InstanceBlueprint('vrp', 500, 'R', 'R', '0', 'inter', 1, 9, 70, 1, n_depots=1)





