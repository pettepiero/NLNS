import numpy as np
import torch
import torch.optim as optim
import csv
import os
from search import destroy_instances
from copy import deepcopy
import logging
import datetime
from search_batch import lns_batch_search
import repair
import main
from vrp.data_utils import create_dataset, save_dataset_pkl, read_instances_pkl, save_dataset_vrplib, read_instance_mdvrp
from search import LnsOperatorPair
from tqdm import tqdm, trange
from pathlib import Path
from plot.plot import plot_instance
import pickle


def save_model_info(config):
    filepath = "./list_trained_models.csv"
    header = [
     "Run_ID",
     "Instance_blueprint",
     "nb_train_batches",
     "nb_batches_training_set",
     "test_size",
     "actor_lr",
     "batch_size",
    ] 
    run_id = os.path.basename(config.output_path)
    row = [
        run_id,
        config.instance_blueprint,
        config.nb_train_batches,
        config.nb_batches_training_set,
        config.test_size,
        config.actor_lr,
        config.batch_size,
    ]
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as f:
       writer = csv.writer(f)
       if not file_exists:
           writer.writerow(header)
       writer.writerow(row)


def train_nlns(actor, critic, run_id, config):
    save_model_info(config)
    rng = np.random.default_rng(config.seed)
    batch_size = config.batch_size

    if not config.load_dataset:
        logging.info("Generating training data...")
        # Create training and validation set. The initial solutions are created greedily
        training_set = create_dataset(size=batch_size * config.nb_batches_training_set, config=config, create_solution=True, use_cost_memory=False, seed=config.seed)
        logging.info("Generating validation data...")
        validation_instances = create_dataset(size=config.valid_size, config=config, seed=config.validation_seed, create_solution=True)

        if config.save_dataset:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if config.dataset_format == 'pkl':
                save_dataset_pkl(training_set, f'./datasets/pkl/{now_str}_train.pkl')
                save_dataset_pkl(validation_instances, f'./datasets/pkl/{now_str}_val.pkl')
            elif config.dataset_format == 'vrplib':
                save_dataset_vrplib(instances=training_set, folder=f'./datasets/vrplib/{now_str}_train/', start_index=1) 
                save_dataset_vrplib(instances=validation_instances, folder=f'./datasets/vrplib/{now_str}_val/', start_index=1) 
            else:
                raise ValueError(f"Unknown dataset_format option: {config.dataset_format}")
#################################################################
        # reading dataset from dir or single pkl file
    else:
        assert config.train_filepath is not None
        assert config.val_filepath is not None
        assert os.path.exists(config.train_filepath)
        assert os.path.exists(config.val_filepath)
       
        # if from directory containing multiple files         
        if os.path.isdir(config.train_filepath):
            # check that there are files that end with 'mdvrp' or 'vrp'
            train_instances = [ins for ins in os.listdir(config.train_filepath) if os.path.isfile(os.path.join(config.train_filepath, ins))]
            train_instances = [ins for ins in train_instances if os.path.splitext(ins)[1] in ['.mdvrp', 'vrp']]
            print(f"Found {len(train_instances)} train instances")
            assert len(train_instances) == config.batch_size * config.nb_batches_training_set
            val_instances = [ins for ins in os.listdir(config.val_filepath) if os.path.isfile(os.path.join(config.val_filepath, ins))]
            validation_instances = [ins for ins in val_instances if os.path.splitext(ins)[1] in ['.mdvrp', 'vrp']]
            print(f"Found {len(val_instances)} val instances")
            #convert to mdvrpinstance list
            print("Converting instances from files to VRPInstance/MDVRPInstance list...")
            training_set = []
            validation_instances = []
            if config.problem_type == 'mdvrp':
                for el in tqdm(train_instances):
                    instance = read_instance_mdvrp(os.path.join(config.train_filepath, el))
                    instance.create_initial_solution()
                    training_set.append(instance)
                for el in tqdm(val_instances):
                    instance = read_instance_mdvrp(os.path.join(config.val_filepath, el))
                    instance.create_initial_solution()
                    validation_instances.append(instance)
            elif config.problem_type == 'vrp':
                for el in tqdm(train_instances):
                    instance = read_instance_vrp(os.path.join(config.train_filepath, el))
                    instance.create_initial_solution()
                    training_set.append(instance)
                for el in tqdm(val_instances):
                    instance = read_instance_vrp(os.path.join(config.val_filepath, el))
                    instance.create_initial_solution()
                    validation_instances.append(instance)
            else:
                raise ValueError('Problem in config.problem_type: {config.problem_type}')
            print("...done")
            assert len(val_instances) % config.lns_batch_size == 0
        
        # if from single pkl file
        elif os.path.splitext(config.train_filepath)[1] == '.pkl' and os.path.splitext(config.val_filepath)[1] == '.pkl':
            with open(config.train_filepath, "rb") as f:
                training_set = pickle.load(f)
            with open(config.val_filepath, "rb") as f:
                validation_instances = pickle.load(f)
#############################################################################################àààà
    actor_optim = optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor.train()
    critic_optim = optim.Adam(critic.parameters(), lr=config.critic_lr)
    critic.train()

    losses_actor, rewards, diversity_values, losses_critic = [], [], [], []
    # save csv files with losses and rewards
    metrics_dir = Path(getattr(config, "metrics_dir", Path(config.output_path) / "metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)  
    # Allow user-defined file paths on config; otherwise default names under metrics_dir
    summary_path = Path(getattr(config, "summary_path", metrics_dir / f"summary_every_250_{run_id}.csv"))
    if not summary_path.exists():
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "batch_idx", "mean_reward_250", "mean_actor_loss_250", "mean_critic_loss_250"])

    incumbent_costs = np.inf
    start_time = datetime.datetime.now()

    logging.info("Starting training...")
    
    for batch_idx in trange(1, config.nb_train_batches + 1):
    #for batch_idx in range(1, config.nb_train_batches + 1):
        # Get a batch of training instances from the training set. Training instances are generated in advance, because
        # generating them is expensive.
        training_set_batch_idx = batch_idx % config.nb_batches_training_set
        tr_instances = [deepcopy(instance) for instance in
                        training_set[training_set_batch_idx * batch_size: (training_set_batch_idx + 1) * batch_size]]

        # Destroy and repair the set of instances
        destroy_instances(rng, tr_instances, config.lns_destruction, config.lns_destruction_p)
        costs_destroyed = [instance.get_costs_incomplete() for instance in tr_instances]
        tour_indices, tour_logp, critic_est = repair.repair(tr_instances, actor, config, critic, rng)
        costs_repaired = [instance.get_costs() for instance in tr_instances]
        # Reward/Advantage computation
        reward = np.array(costs_repaired) - np.array(costs_destroyed)
        reward = torch.from_numpy(reward).float().to(config.device)
        advantage = reward - critic_est

        # Actor loss computation and backpropagation
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_optim.step()

        # Critic loss computation and backpropagation
        critic_loss = torch.mean(advantage ** 2)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_optim.step()

        rewards.append(torch.mean(reward.detach()).item())
        losses_actor.append(torch.mean(actor_loss.detach()).item())
        losses_critic.append(torch.mean(critic_loss.detach()).item())

        # Replace the solution of the training set instances with the new created solutions
        for i in range(batch_size):
            training_set[training_set_batch_idx * batch_size + i] = tr_instances[i]

        # Log performance every 250 batches
        if batch_idx % 250 == 0 and batch_idx > 0:
            mean_loss = np.mean(losses_actor[-250:])
            mean_critic_loss = np.mean(losses_critic[-250:])
            mean_reward = np.mean(rewards[-250:])

            logging.info(
                f'Batch {batch_idx}/{config.nb_train_batches}, repair costs (reward): {mean_reward:2.3f}, loss: {mean_loss:2.6f} , critic_loss: {mean_critic_loss:2.6f}')
            
            now = datetime.datetime.now().isoformat()
            with summary_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([now, batch_idx, mean_reward, mean_loss, mean_critic_loss])

        # Evaluate and save model every 5000 batches
        if batch_idx % 5000 == 0 or batch_idx == config.nb_train_batches:
            mean_costs = lns_validation_search(validation_instances, actor, config, rng)
            model_data = {
                'parameters': actor.state_dict(),
                'model_name': "VrpActorModel",
                'destroy_operation': config.lns_destruction,
                'p_destruction': config.lns_destruction_p,
                'code_version': main.VERSION
            }

            if config.split_delivery:
                problem_type = "SD"
            else:
                problem_type = "C"
            torch.save(model_data, os.path.join(config.output_path, "models",
                                                "model_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                      config.instance_blueprint,
                                                                                      config.lns_destruction,
                                                                                      config.lns_destruction_p,
                                                                                      run_id)))
            if mean_costs < incumbent_costs:
                incumbent_costs = mean_costs
                incumbent_model_path = os.path.join(config.output_path, "models",
                                                    "model_incumbent_{0}_{1}_{2}_{3}_{4}.pt".format(problem_type,
                                                                                                    config.instance_blueprint,
                                                                                                    config.lns_destruction,
                                                                                                    config.lns_destruction_p,
                                                                                                    run_id))
                torch.save(model_data, incumbent_model_path)

            runtime = (datetime.datetime.now() - start_time)
            logging.info(
                f"Validation (Batch {batch_idx}) Costs: {mean_costs:.3f} ({incumbent_costs:.3f}) Runtime: {runtime}")
    return incumbent_model_path


def lns_validation_search(validation_instances, actor, config, rng):
    validation_instances_copies = [deepcopy(instance) for instance in validation_instances]
    actor.eval()
    operation = LnsOperatorPair(actor, config.lns_destruction, config.lns_destruction_p)
    costs, _ = lns_batch_search(validation_instances_copies, config.lns_max_iterations,
                                config.lns_timelimit_validation, [operation], config, rng)
    actor.train()
    return np.mean(costs)
