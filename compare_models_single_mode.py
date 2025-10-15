import argparse
import os
from pathlib import Path
from tqdm import tqdm
from vrp.data_utils import save_dataset_pkl, read_instance_mdvrp, read_instances_pkl
import subprocess
import datetime
import numpy as np
import logging

def check_valid_dir(directory: Path) -> bool:
    """
    Returns True if provided directory contains at least one file
    ending with '.mdvrp'
    """
    assert os.path.exists(directory), f"Provided path {directory} doesn't exist"
    assert os.path.isdir(directory), f"Provided path {directory} is not a folder"
    inst_list = os.listdir(directory)
    inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.mdvrp']
    assert len(inst_list) > 0, f"Provided path {len(inst_list)} doesn't contain files with '.mdvrp' extension."

    return len(inst_list) > 0

def get_dir_filenames(directory: Path) -> list:
    if check_valid_dir(directory):
        inst_list = os.listdir(directory)
        inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.mdvrp']
        assert len(inst_list) > 0, f"Provided path {len(inst_list)} doesn't contain files with '.mdvrp' extension."
        logging.debug(f"DEBUG: Provided path contains {len(inst_list)} files with '.mdvrp' extension.")
        return inst_list
    else:
        raise ValueError

def read_dir(directory: Path, max_num_instances: int) -> Path:
    inst_dir = get_dir_filenames(directory)
    if max_num_instances is not None:
        n_instances = min(max_num_instances, len(inst_list))
        inst_list = random.shuffle(inst_list)[:n_instances]
    else:
        n_instances = len(inst_list)
    logging.debug(f"DEBUG: Selecting {n_instances}/{len(inst_list)} random instances from provided directory.") 
    dataset = []
    logging.debug(f"Reading instances and creating dataset...")
    for inst in tqdm(inst_list):
        instance = read_instance_mdvrp(os.path.join(directory, inst))
        dataset.append(instance)
    logging.debug(f"...done")

    # create pkl file for NLNS
    pkl_filepath = Path(directory) / 'dataset.pkl'
    save_dataset_pkl(instances=dataset, output_path=pkl_filepath)

    return pkl_filepath, n_instances


def read_pkl(filepath: Path, max_num_instances: int) -> Path:
    assert os.path.exists(filepath), f"Provided path {filepath} doesn't exist"
    assert os.path.isfile(filepath), f"Provided path {filepath} is not a file"
    dataset = read_instances_pkl(pkl_name)
    assert len(dataset) > 0, f"Provided file doesn't contain instances."
    logging.debug(f"DEBUG: Provided file contains {len(dataset)} mdvrp instances.")
    if max_num_instances is not None:
        if len(dataset) > max_num_instances:
            dataset = random.shuffle(dataset)[:max_num_instances]
            logging.debug(f"DEBUG: Selecting {max_num_instances}/{len(dataset)} random instances from provided pkl file.") 
            pkl_filepath = filepath.with_name(filepath.stem + '_cut.' + filepath.suffix)
            save_dataset_pkl(instances=dataset, output_path=pkl_filepath)
        else:
            logging.debug(f"DEBUG: Selecting all {len(dataset)} instances of pkl file.")
            pkl_filepath = filepath

    return pkl_filepath, len(dataset) 

#read folder with data to test models
ap = argparse.ArgumentParser()
ap.add_argument('--mode', type=str, default='read_dir', choices=['read_dir', 'read_pkl'], required=True, help="Modality of data reading. Either read directory or read pkl file")
ap.add_argument('--path', '-p', type=Path, required=True, help="Path of data dir or pkl file")
#ap.add_argument('--data_folder', '-f', type=Path, help="Folder containing instances to test models on", required=True)
ap.add_argument('--nlns_max_time_per_instance', type=int, default=30, help="Maximum solve time per instance by NLNS model. Default 30s")
ap.add_argument('--pyvrp_max_time_per_instance', type=int, default=30, help="Maximum solve time per instance by PyVRP model. Default 30s")
ap.add_argument('--max_num_instances', '-n', type=int, default=None, help="Maximum number of instances to solve. Default: all in the directory.")
ap.add_argument('--nlns_model', type=str, default=None, help="NLNS model to test. Provide run number, e.g. 'run_17.9.2025_16354', or full model path if --full_model_path is set to true. See list_trained_models.csv for a list of trained NLNS models.", required=True)
ap.add_argument('--full_model_path', default=False, action='store_true', help="Set to True if nlns_model is the full model path")
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to run on.")

run_id = np.random.randint(10000, 99999)
output_path = os.getcwd()
now = datetime.datetime.now()
output_path = os.path.join(output_path, "runs", f"run_{now.day}.{now.month}.{now.year}_{run_id}")
os.makedirs(os.path.join(output_path, "solutions"))
os.makedirs(os.path.join(output_path, "models"))
os.makedirs(os.path.join(output_path, "search"))
args = ap.parse_args()
log_filename = os.path.join(output_path, f"log_{run_id}.txt")
logging.basicConfig(filename=log_filename, level=logging.DEBUG)

print(f"Running compare_models_single_mode.py with run_id {run_id}")
print(f"Log of this execution is being written to {log_filename}")

logging.debug(f"Log of compare_models_single_mode.py run on {now.day}/{now.month}/{now.year} at {now.hour}:{now.minute}:{now.second}")

logging.debug(f"Parsed args:")

for el in vars(args):
    logging.debug(f"{el}")

# Read dataset
if args.mode == 'read_dir':
    #pkl_file, num_instances = read_dir(args.path, args.max_num_instances)
    valid_dir = check_valid_dir(args.path)
    if not valid_dir:
        raise ValueError(f"Invalid directory {args.path}")

elif args.mode == 'read_pkl':
    raise NotImplementedError
    pkl_file, num_instances = read_pkl(args.path, args.max_num_instances)

# Run NLNS
if not args.full_model_path:
    model_path = Path('./runs/') / args.nlns_model / 'models'
    models = list(model_path.glob("model_incumbent*.pt"))
    assert len(models) <= 1, f"Too many possible models found. Use full model specification"
    assert len(models) > 0, f"Did not find any models in {model_path}"
    full_model_path = models[0]
else:
    assert os.path.exists(args.full_model_path), f"Error: full model path {args.full_model_path} not found"
    full_model_path = args.nlns_model

assert os.path.exists(full_model_path), f"Provided model_path doesn't exists"

models_dir = full_model_path.parent

logging.debug(f"\n**********************************************\nCalling NLNS model to run on batch:\n")
# execute NLNS batch eval

cmd_nlns = [
    "python3",          "main.py",
    "--mode",           "eval_single",
    "--model_path",     full_model_path,
    "--instance_path",  args.path,
    "--lns_batch_size", "2",
    "--lns_timelimit",  str(args.nlns_max_time_per_instance),
    "--problem_type",   "mdvrp",
    "--device",         args.device,
    "--output_path",    output_path,
    ]

logging.debug(f"NLNS command: {cmd_nlns}")

subprocess.run(cmd_nlns, check=True)

logging.debug(f"Written NLNS objective traces to {models_dir}/objective_trace_inst_INST_NUM.mdvrp.csv")

logging.debug(f"\n*****************************************************")
logging.debug("Running PyVRP model...\n")

cmd_pyvrp = [
    "python3",          "pyvrp_model.py",
    "--mode",           "eval_batch",
    "--dir_path",       args.path,
    "--output_dir",     output_path,
    "--max_time",       str(args.pyvrp_max_time_per_instance),
    ]

logging.debug(f"PyVRP command: {cmd_pyvrp}")

subprocess.run(cmd_pyvrp, check=True)

#summarize metrics
nlns_costs = []
nlns_filepath = os.path.join(output_path, "search", "nlns_batch_search_results.txt")
with open(nlns_filepath, 'r') as f:
    nlns_costs = f.read().splitlines()
#round nlns_costs like the pyvrp_ones
nlns_costs = [
    f"{idx},{round(float(cost))}" for idx, cost in (item.split(",") for item in nlns_costs)
    ]
pyvrp_costs = []
pyvrp_filepath = os.path.join(output_path, "search", "pyvrp_eval_batch.txt")
with open(pyvrp_filepath, 'r') as f:
    pyvrp_costs = f.read().splitlines()

logging.debug(f"Saved NLNS costs to: {nlns_filepath}")
logging.debug(f"Saved PyVRP costs to: {pyvrp_filepath}")

#costs_gap = (nlns_costs - pyvrp_costs)/pyvrp_costs
costs_gap = [
    (float(nlns) - float(pyvrp))/float(pyvrp)
    for (_, nlns), (_, pyvrp) in zip(
        (item.split(",") for item in nlns_costs),
        (item.split(",") for item in pyvrp_costs)
    )
]

avg_costs_gap = sum(costs_gap) / len(costs_gap)

logging.debug(f"\n\nAverage costs gap: {avg_costs_gap}")

print(f"Saved log of execution to {log_filename}")
