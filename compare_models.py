import argparse
import os
from pathlib import Path
from tqdm import tqdm
from vrp.data_utils import save_dataset_pkl, read_instance_mdvrp, read_instances_pkl
import subprocess

def read_dir(directory: Path, max_num_instances: int) -> Path:
    assert os.path.exists(directory), f"Provided path {directory} doesn't exist"
    assert os.path.isdir(directory), f"Provided path {directory} is not a folder"

    inst_list = os.listdir(directory)
    inst_list = [ins for ins in inst_list if os.path.splitext(ins)[1] == '.mdvrp']
    assert len(inst_list) > 0, f"Provided path {len(inst_list)} doesn't contain files with '.mdvrp' extension."
    print(f"DEBUG: Provided path contains {len(inst_list)} files with '.mdvrp' extension.")

    if max_num_instances is not None:
        n_instances = min(max_num_instances, len(inst_list))
        inst_list = random.shuffle(inst_list)[:n_instances]
    else:
        n_instances = len(inst_list)
    print(f"DEBUG: Selecting {n_instances}/{len(inst_list)} random instances from provided directory.") 
    dataset = []
    print(f"Reading instances and creating dataset...")
    for inst in tqdm(inst_list):
        instance = read_instance_mdvrp(os.path.join(directory, inst))
        dataset.append(instance)
    print(f"...done")

    # create pkl file for NLNS
    pkl_filepath = Path(directory) / 'dataset.pkl'
    save_dataset_pkl(instances=dataset, output_path=pkl_filepath)

    return pkl_filepath 


def read_pkl(filepath: Path, max_num_instances: int) -> Path:
    assert os.path.exists(filepath), f"Provided path {filepath} doesn't exist"
    assert os.path.isfile(filepath), f"Provided path {filepath} is not a file"
    dataset = read_instances_pkl(pkl_name)
    assert len(dataset) > 0, f"Provided file doesn't contain instances."
    print(f"DEBUG: Provided file contains {len(dataset)} mdvrp instances.")
    if max_num_instances is not None:
        if len(dataset) > max_num_instances:
            dataset = random.shuffle(dataset)[:max_num_instances]
            print(f"DEBUG: Selecting {max_num_instances}/{len(dataset)} random instances from provided pkl file.") 
            pkl_filepath = filepath.with_name(filepath.stem + '_cut.' + filepath.suffix)
            save_dataset_pkl(instances=dataset, output_path=pkl_filepath)
        else:
            print(f"DEBUG: Selecting all {len(dataset)} instances of pkl file.")
            pkl_filepath = filepath

    return pkl_filepath 

#read folder with data to test models
ap = argparse.ArgumentParser()
ap.add_argument('--mode', type=str, default='read_dir', choices=['read_dir', 'read_pkl'], required=True, help="Modality of data reading. Either read directory or read pkl file")
ap.add_argument('--path', '-p', type=Path, required=True, help="Path of data dir or pkl file")

#ap.add_argument('--data_folder', '-f', type=Path, help="Folder containing instances to test models on", required=True)
ap.add_argument('--max_time', '-t', type=int, default=30, help="Maximum solve time per instance. Default 30s")
ap.add_argument('--max_num_instances', '-n', type=int, default=None, help="Maximum number of instances to solve. Default: all in the directory.")
ap.add_argument('--nlns_model', type=str, default=None, help="NLNS model to test. Provide run number, e.g. 'run_17.9.2025_16354', or full model path if --full_model_path is set to true. See list_trained_models.csv for a list of trained NLNS models.", required=True)
ap.add_argument('--full_model_path', default=False, action='store_true', help="Set to True if nlns_model is the full model path")

args = ap.parse_args()
print(f"DEBUG: args: {vars(args)}")

# Read dataset
if args.mode == 'read_dir':
    pkl_file = read_dir(args.path, args.max_num_instances)
    
elif args.mode == 'read_pkl':
    pkl_file = read_pkl(args.path, args.max_num_instances)


# Run NLNS
if not args.full_model_path:
    model_path = Path('./runs/') / args.nlns_model / 'models'
    models = list(model_path.glob("model_incumbent*.pt"))
    assert len(models) <= 1, f"Too many possible models found. Use full model specification"
    assert len(models) > 0, f"Did not find any models in {model_path}"
    full_model_path = models[0]
else:
    assert os.path.exists(args.full_model_path), f"Error: full model path {args.full_model_path} not found"
    full_model_path = full_model_path
assert os.path.exists(model_path), f"Provided model_path doesn't exists"

print(f"\n**********************************************\nCalling NLNS model to run on batch:\n")
# execute NLNS batch eval

cmd = [
    "python3",  "main.py",
    "--mode",   "eval_batch",
    "--model_path",     full_model_path,
    "--instance_path",  pkl_file,
    "--lns_batch_size", "2",
    "--lns_timelimit",  str(args.max_time),
    "--problem_type",   "mdvrp",
    ]

subprocess.run(cmd, check=True)


#summarize metrics


