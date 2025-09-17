# NOTE: if this code fails for unhashable numpy type, add following line in vrplib parse_section function:

#        if name == "vehicles_depot":
#            data = np.array([row[0] for row in rows])
#        else:
#            data = np.array([row[1:] for row in rows])



import argparse
from pyvrp import Model, read
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution, plot_coordinates
import matplotlib.pyplot as plt
import numpy as np
from vrplib.read import read_instance

parser = argparse.ArgumentParser(description="PyVRP model execution")
parser.add_argument(
    "--instance_path",
    default=None,
    type=str,
    help="Path to one VRPLIB file",
)
parser.add_argument(
    "--max_time",
    type=float,
    default=10.0,
    help="Max runtime per instance in seconds (float).",
)
parser.add_argument(
    "--plot_solution", "--plot-solution",
    dest='plot_solution',
    action='store_true',
    help="Plot instance solution.",
)
args = parser.parse_args()

data = read_instance(args.instance_path)
assert isinstance(data['capacity'], int)
m = Model()
num_depots = len(data['depot'])
depots = []
for d in data['depot']:
    depot = m.add_depot(x=data['node_coord'][d][0], y=data['node_coord'][d][1])
    depots.append(depot)

keys, counts = np.unique(data['vehicles_depot'], return_counts=True)
keys = keys - 1
keys = keys.tolist()
counts = counts.tolist()
depot_num_vehicles =  dict(zip(keys, counts))

for i, d in enumerate(data['depot']):
    m.add_vehicle_type(
        num_available   = depot_num_vehicles[int(d)],
        capacity        = data['capacity'],
        start_depot     = depots[i],
        end_depot       = depots[i],
    )

clients = [
    m.add_client(
        x=int(data['node_coord'][idx][0]),
        y=int(data['node_coord'][idx][1]),
        delivery=int(data['demand'][idx]),
    )
    for idx in range(num_depots, len(data['node_coord']))
]

locations = depots + clients

for frm_idx, frm in enumerate(locations):
    for to_idx, to in enumerate(locations):
        #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
        distance = np.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2) 
        m.add_edge(frm, to, distance=distance)

result = m.solve(stop=MaxRuntime(args.max_time), display=True)

print(result)

if args.plot_solution:
    plot_solution(result.best, m.data())

_, ax = plt.subplots(figsize=(8, 8))
plot_solution(result.best, m.data(), ax=ax)
plt.show()
