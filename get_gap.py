import pandas as pd
import os
import argparse
from pathlib import Path
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Get gap of compare models runs")
    parser.add_argument('--run_folder', '-f', type=str, required=True, help="Relative path to run's folder")

    args = parser.parse_args()
    assert os.path.exists(args.run_folder), f"Argument {args.run_folder} is not a valid path"
    assert os.path.isdir(args.run_folder), f"Argument {args.run_folder} is not a folder"

    pyvrp_path = Path(args.run_folder) / 'search' / 'pyvrp_eval_batch.txt'
    nlns_path = Path(args.run_folder) / 'search' / 'nlns_batch_search_results.txt'

    pyvrp_results = pd.read_csv(pyvrp_path, names=['instance_id', 'pyvrp_cost'])
    nlns_results = pd.read_csv(nlns_path, names=['instance_id', 'nlns_cost'])

    results = pd.merge(pyvrp_results, nlns_results, on='instance_id')
    results['gap_perc'] = (results['nlns_cost'] - results['pyvrp_cost']) / results['pyvrp_cost']
    results['gap_perc'] = round(results['gap_perc']*100, 1)

    print(f"results:")
    print(results)

    print(f"Average gap on {len(results['instance_id'])} instances: {round(np.mean(results['gap_perc']), 1)}%")
