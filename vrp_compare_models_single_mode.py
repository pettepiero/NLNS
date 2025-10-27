import argparse
import os
from pathlib import Path
from tqdm import tqdm
from vrp.data_utils import save_dataset_pkl, read_instance_mdvrp, read_instances_pkl
import subprocess
import datetime
import numpy as np
import logging
import wandb

def check_valid_dir(directory: Path) -> bool:
    """
    Returns True if provided directory contains at least one file
    ending with '.mdvrp'
    """
    assert os.path.exists(directory), f"Provided path {directory} doesn't exist"
    assert os.path.isdir(directory), f"Provided path {directory} is not a folder"
    inst_list = os.listdir(directory)
    inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.vrp']
    assert len(inst_list) > 0, f"Provided path {len(inst_list)} doesn't contain files with '.vrp' extension."

    return len(inst_list) > 0

def get_dir_filenames(directory: Path) -> list:
    if check_valid_dir(directory):
        inst_list = os.listdir(directory)
        inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.vrp']
        assert len(inst_list) > 0, f"Provided path {len(inst_list)} doesn't contain files with '.vrp' extension."
        logging.debug(f"DEBUG: Provided path contains {len(inst_list)} files with '.vrp' extension.")
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
ap.add_argument('--darp_max_time_per_instance', type=int, default=30, help="Maximum solve time per instance by darp model. Default 30s")
ap.add_argument('--pyvrp_max_time_per_instance', type=int, default=30, help="Maximum solve time per instance by PyVRP model. Default 30s")
ap.add_argument('--max_num_instances', '-n', type=int, default=None, help="Maximum number of instances to solve. Default: all in the directory.")
ap.add_argument('--nlns_model', type=str, default=None, help="NLNS model to test. Provide run number, e.g. 'run_17.9.2025_16354', or full model path if --full_model_path is set to true. See list_trained_models.csv for a list of trained NLNS models.", required=True)
ap.add_argument('--full_model_path', default=False, action='store_true', help="Set to True if nlns_model is the full model path")
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help="Device to run on.")

# LNS single instance search parameters
ap.add_argument('--lns_t_max', default=1000, type=int, help="Maximum reheating temperature for Simulated Annealing of single instance search")
ap.add_argument('--lns_t_min', default=10, type=int, help="Minimum reheating temperature for Simulated Annealing of single instance search")
ap.add_argument('--lns_reheating_nb', default=5, type=int, help="Number of reheating operations for Simulated Annealing of single instance search")
ap.add_argument('--lns_Z_param', default=0.8, type=float, help="What percentage of the search focuses on generating neighbouring solution. See NLNS paper (expressed between 0 and 1)")
ap.add_argument('--lns_nb_cpus', default=1, type=int, help="How many instances compose the search batch B in single instance search") 
ap.add_argument('--wandb', action='store_true', help="If set, sends logs to wandb")

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

print(f"Running compare_models_single_mode.py with run_id {run_id} and logging to wandb 'compare_models_single_mode' project.")
print(f"Log of this execution is being written to {log_filename}")

if args.wandb:
    # wandb logging
    wandb.init(
        project="compare_models_single_mode",
        id=str(run_id),
        tags=["compare"],
        config=args,
    )

logging.debug(f"Log of compare_models_single_mode.py run on {now.day}/{now.month}/{now.year} at {now.hour}:{now.minute}:{now.second}")
logging.debug(f"Running compare_models_single_mode.py with run_id {run_id} and logging to wandb 'compare_models_single_mode' project.")
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

#darp_output_path = os.path.join(output_path, f"darp_results_{run_id}.txt")
#print(f"\n ************************************************** \n")
#print(f"EXECUTING dial-a-ride ALNS...")
## Run DARP
#cmd_darp = [
#    "python3",              "/home/pettena/dial-a-ride/cmdvrp.py",
#    "--mode",               "batch",
#    "--problem_type",       "mdvrp",
#    "--dir",                args.path,
#    "--stop_criterion",     "runtime",
#    "--max_time",           str(args.darp_max_time_per_instance),
#    "--output_path",        darp_output_path,
#]
#logging.debug(f"Calling DARP (cmdvrp) with this command:")
#logging.debug(cmd_darp)
#subprocess.run(cmd_darp, check=True)
#print(f"... Done\n")
#
print(f"\n ************************************************** \n")
print(f"EXECUTING NLNS models...")
# Run NLNS
if not args.full_model_path:
    if args.nlns_model == 'trained_models/cmdvrp/MD_8/':
        full_model_path = Path(args.nlns_model)
        logging.debug(f"Using models in folder {args.nlns_model}")
    else:
        logging.debug(f"Trying to get model from path: {args.nlns_model}")
# if exists, as is:
        if os.path.exists(args.nlns_model):
        # if file -> single model
            if os.path.isfile(args.nlns_model):
                full_model_path = Path(args.nlns_model)
                models_dir = full_model_path.parent
            # else -> models folder
            elif os.path.isdir(args.nlns_model):
                full_model_path = Path(args.nlns_model)
                models_dir = full_model_path
            else:
                raise ValueError
        else: # try adding 'runs' and 'models' 
            model_path = Path('./runs/') / args.nlns_model / 'models'
            models = list(model_path.glob("model_incumbent*.pt"))
            assert len(models) <= 1, f"Too many possible models found. Use full model specification"
            assert len(models) > 0, f"Did not find any models in {model_path}"
            full_model_path = models[0]
            models_dir = full_model_path.parent
else:
    assert os.path.exists(args.full_model_path), f"Error: full model path {args.full_model_path} not found"
    full_model_path = args.nlns_model
    models_dir = full_model_path.parent

assert os.path.exists(full_model_path), f"Provided model_path doesn't exists"


logging.debug(f"\n**********************************************\nCalling NLNS model to run on batch:\n")
# execute NLNS batch eval

cmd_nlns = [
    "python3",              "main.py",
    "--mode",               "eval_single",
    "--model_path",         full_model_path,
    "--instance_path",      args.path,
    "--lns_batch_size",     "2",
    "--lns_timelimit",      str(args.nlns_max_time_per_instance),
    "--problem_type",       "vrp",
    "--device",             args.device,
    "--output_path",        output_path,
    "--lns_t_max",          str(args.lns_t_max),
    "--lns_t_min",          str(args.lns_t_min),
    "--lns_reheating_nb",   str(args.lns_reheating_nb),
    "--lns_Z_param",        str(args.lns_Z_param),
    "--lns_nb_cpus",        str(args.lns_nb_cpus),
    ]

logging.debug(f"NLNS command: {cmd_nlns}")

subprocess.run(cmd_nlns, check=True)

logging.debug(f"Written NLNS objective traces to {models_dir}/objective_trace_inst_INST_NUM.mdvrp.csv")
print(f"... Done\n")

# log objective traces

logging.debug(f"\n*****************************************************")

print(f"\n ************************************************** \n")
print(f"EXECUTING PyVRP models...")
# first check if results are already available
pyvrp_filepath = f'pyvrp_runs/{args.path}_{args.pyvrp_max_time_per_instance}.csv'
# if available, read those directly
found_pyvrp_file = False
logging.debug(f"Looking for already solved instances in: {pyvrp_filepath}")
if os.path.isfile(pyvrp_filepath):
    found_pyvrp_file = True
    logging.debug(f"Found already solved instances: {pyvrp_filepath}")

if not found_pyvrp_file:
    # if not available, call model
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

print(f"... Done\n")
print(f"************************************************")
print(f"Summarizing metrics:")
#summarize metrics
darp_costs = []
#with open(darp_output_path, 'r') as f:
#    darp_costs = f.read().splitlines()
#    darp_costs = darp_costs[1:]
#darp_costs = [
#    f"{idx}, {round(float(cost))}" for idx, _, cost, _ in (item.split(",") for item in darp_costs)
#    ]
nlns_costs = []
nlns_filepath = os.path.join(output_path, "search", "nlns_batch_search_results.txt")
with open(nlns_filepath, 'r') as f:
    nlns_costs = f.read().splitlines()
#round nlns_costs like the pyvrp_ones
nlns_costs = [
    f"{idx},{round(float(cost))}" for idx, cost in (item.split(",") for item in nlns_costs)
    ]
pyvrp_costs = []
if not found_pyvrp_file:
    pyvrp_filepath = os.path.join(output_path, "search", "pyvrp_eval_batch.txt")
with open(pyvrp_filepath, 'r') as f:
    pyvrp_costs = f.read().splitlines()
    assert len(pyvrp_costs) == len(nlns_costs)
    #assert len(darp_costs) == len(nlns_costs)

#logging.debug(f"Saved darp costs to: {darp_output_path}")
logging.debug(f"Saved NLNS costs to: {nlns_filepath}")
logging.debug(f"Saved PyVRP costs to: {pyvrp_filepath}")
#print(f"Saved darp costs to: {darp_output_path}")
print(f"Saved NLNS costs to: {nlns_filepath}")
print(f"Saved PyVRP costs to: {pyvrp_filepath}")

#darp_costs_gap = [
#    (float(darp) - float(pyvrp))/float(pyvrp)
#    for (_, darp), (_, pyvrp) in zip(
#        (item.split(",") for item in darp_costs),
#        (item.split(",") for item in pyvrp_costs)
#    )
#]

#avg_darp_costs_gap = sum(darp_costs_gap) / len(darp_costs_gap)
#
#logging.debug(f"\n\nAverage darp costs gap: {avg_darp_costs_gap}")
#print(f"\n\nAverage darp costs gap: {avg_darp_costs_gap}")

#costs_gap = (nlns_costs - pyvrp_costs)/pyvrp_costs
nlns_costs_gap = [
    (float(nlns) - float(pyvrp))/float(pyvrp)
    for (_, nlns), (_, pyvrp) in zip(
        (item.split(",") for item in nlns_costs),
        (item.split(",") for item in pyvrp_costs)
    )
]

avg_nlns_costs_gap = sum(nlns_costs_gap) / len(nlns_costs_gap)

logging.debug(f"\n\nAverage NLNS costs gap: {avg_nlns_costs_gap}")
print(f"\n\nAverage NLNS costs gap: {avg_nlns_costs_gap}")

print(f"Saved log of execution to {log_filename}")
