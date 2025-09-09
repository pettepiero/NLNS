import pyvrp
import argparse
from vrp.data_utils import read_instances_pkl, convert_data_notation
import os

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyVRP model execution')
    parser.add_argument('--mode', default='eval_batch', type=str, choices=['eval_single', 'eval_batch'])
    parser.add_argument('--dataset_dir', default=None, type=str) 
    parser.add_argument('--dataset_format', default=None, type=str, choices=['pkl', 'vrplib'])

    args = parser.parse_args()

    assert os.path.exists(args.dataset_dir)
    if args.dataset_format == 'pkl':
        pkl_dataset = read_instances_pkl(args.dataset_dir)
        dataset = save_dataset_vrplib(pkl_dataset, folder='args.dataset_dir' + '/vrplib/') 
        raise NotImplementedError
    elif args.dataset_format == 'vrplib':
        dirname = args.dataset_dir
        dataset = [pyvrp.read(os.path.join(dirname, filename)) for filename in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, filename))]
    
    
    print(dataset[0])

    #m = pyvrp.Model()
    #res  = m.solve(stop=pyvrp.stop.MaxRuntime(1))
    res = pyvrp.solve(dataset[0], stop=pyvrp.stop.MaxRuntime(1))
    print(res)
