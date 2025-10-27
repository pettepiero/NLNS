from vrp.data_utils import mdvrp_to_plot_solution, read_instance
from pyvrp.plotting import plot_solution
from pyvrp import read as pyvrp_read
import pickle
import os
from pathlib import Path
import argparse
from search_single import has_vehicles_line

def load_instance(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def has_vehicles_line(instance_path: str) -> bool:
    """
    Determines wether the instance_path has line 'VEHICLES : INF' or not
    """
    lines = []
    with open(instance_path, 'r') as f:
        lines = f.readlines() 
    for line in lines:
        if line == 'VEHICLES : INF\n':
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Plot evolution of instances")

    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--instance_path', type=str, required=True)
   
    args = parser.parse_args() 

    instances_list = os.listdir(args.dir)   
    assert len(instances_list) > 0
    
    instances_list = sorted(Path(args.dir).glob('instance_0_*.pkl'))
    instances_list = [str(el) for el in instances_list]
    
    # read file with problem data
    # if file has 'VEHICLES : INF' line
    data, temp_filename = None, None
    if has_vehicles_line(args.instance_path):
        #  create temp file without line
        temp_filename = f"{args.instance_path}_temp"
        # load data from temp file
        with open(args.instance_path) as f:
            lines = f.readlines()
            lines = [l for l in lines if l != 'VEHICLES : INF\n']
            with open(temp_filename, "w+") as f1:
                f1.writelines(lines)
        data = pyvrp_read(temp_filename)
    else: 
        data = pyvrp_read(args.instance_path) 

    for ins in instances_list:
        instance = read_instance(ins, 0)
        sol = mdvrp_to_plot_solution(instance)
        ins_path = Path(ins)
        plot_solution(sol, data, plot_clients=True, name=ins_path.stem)

    if temp_filename is not None:
        os.remove(temp_filename)



if __name__ == '__main__':
    main() 
       
