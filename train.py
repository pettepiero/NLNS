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
from vrp.data_utils import create_dataset, save_dataset_pkl, read_instances_pkl, save_dataset_vrplib, read_instance_mdvrp, read_instance_vrp, mdvrp_to_plot_solution
from search import LnsOperatorPair
from tqdm import tqdm, trange
from pathlib import Path
from plot.plot import plot_instance
import pickle
import wandb
from pyvrp.plotting import plot_solution
from pyvrp import read as pyvrp_read

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
    #save_model_info(config)
    #t0 = datetime.datetime.now()
    rng = np.random.default_rng(config.seed)
    batch_size = config.batch_size

    if config.wandb:
        # wandb logging
        wandb.init(
            project="cmdvrp-nlns",
            id=str(run_id),
            tags=["training"],
            config=config,
        )
        wandb.define_metric('batch_idx')
        wandb.define_metric('train/*', step_metric='batch_idx')
        wandb.define_metric('tour_stats/', step_metric='batch_idx')
        wandb.define_metric('single_instance/*')

    #t1 = datetime.datetime.now()
    #logging.info(f"Time to initialize wandb: {t1 - t0}") 
    #t0 = t1 
    if not config.load_dataset:
        logging.info("Generating training data...")
        # Create training and validation set. The initial solutions are created greedily
        training_set = create_dataset(size=batch_size * config.nb_batches_training_set, config=config, create_solution=True, use_cost_memory=False, seed=config.seed)
        logging.info("Generating validation data...")
        validation_instances = create_dataset(size=config.valid_size, config=config, seed=config.validation_seed, create_solution=True)

        val_dataset_path, train_dataset_path = None, None

        if config.save_dataset:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if config.dataset_format == 'pkl':
                save_dataset_pkl(training_set, f'./datasets/pkl/{now_str}_train.pkl')
                save_dataset_pkl(validation_instances, f'./datasets/pkl/{now_str}_val.pkl')
            elif config.dataset_format == 'vrplib':
                train_dataset_path = f'./datasets/vrplib/{now_str}_train/'
                val_dataset_path = f'./datasets/vrplib/{now_str}_val/'
                logging.info(f"Saving datasets to {train_dataset_path} and {val_dataset_path}")
                save_dataset_vrplib(instances=training_set, folder=train_dataset_path, start_index=1) 
                save_dataset_vrplib(instances=validation_instances, folder=val_dataset_path, start_index=1) 
            else:
                train_dataset_path = f'./datasets/vrplib/{now_str}_train/'
                val_dataset_path = f'./datasets/vrplib/{now_str}_val/'
                # when in doubt, use vrplib
                print(f"Unknown dataset_format specification: {config.dataset_format}: going to use 'vrplib'")
                config.dataset_format = 'vrplib'
                save_dataset_vrplib(instances=training_set, folder=train_dataset_path, start_index=1) 
                save_dataset_vrplib(instances=validation_instances, folder=val_dataset_path, start_index=1) 
                logging.info(f"Saving datasets to {train_dataset_path} and {val_dataset_path}")

        # plotting initial solutions and printing costs
        if config.plot_solution:
            if config.problem_type == 'mdvrp':
                sol = mdvrp_to_plot_solution(training_set[0])
                data = pyvrp_read(f"{train_dataset_path}inst_00001.mdvrp") 
                plot_solution(solution=sol, data=data, plot_clients=True, name='initial_sol', plot_title=f'{training_set[0].get_costs()}')
            else:
                sol = mdvrp_to_plot_solution(training_set[0])
                data = pyvrp_read(f"{train_dataset_path}inst_00001.vrp") 
                plot_solution(solution=sol, data=data, plot_clients=True, name='initial_sol', plot_title=f'{training_set[0].get_costs()}')


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
            train_instances = [ins for ins in train_instances if os.path.splitext(ins)[1] in ['.mdvrp', '.vrp']]
            print(f"Found {len(train_instances)} train instances")
            if len(train_instances) > config.batch_size * config.nb_batches_training_set:
                #select the first config.batch_size * config.nb_batches_training_set instances
                train_instances = train_instances[:config.batch_size * config.nb_batches_training_set]
            elif len(train_instances) < config.batch_size * config.nb_batches_training_set:
                raise ValueError(f"There are {len(train_instances)} instances in folder but model was expecting batch_size*nb_batches_training_set = {config.batch_size*config.nb_batches_training_set}")

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
                    if config.rand_init_sol:
                        instance.create_initial_solution_random(rng)
                    else:
                        instance.create_initial_solution()
                    training_set.append(instance)
                for el in tqdm(val_instances):
                    instance = read_instance_mdvrp(os.path.join(config.val_filepath, el))
                    if config.rand_init_sol:
                        instance.create_initial_solution_random(rng)
                    else:
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
    #t1 = datetime.datetime.now()
    #logging.info(f"Time to create/read dataset: {t1-t0}")
    #t0 = t1

    if config.scale_rewards:
        # save costs of initial solutions to scale rewards later on
        init_costs = np.ndarray(len(training_set))
        for i, instance in enumerate(training_set):
            init_costs[i] = instance.get_costs(False)

    actor_optim = optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor.train()
    critic_optim = optim.Adam(critic.parameters(), lr=config.critic_lr)
    critic.train()

    log_f = config.log_f
    assert log_f < config.nb_train_batches, f"Asked to log metrics every {log_f} batches but nb_train_batches = {config.nb_train_batches}"

    losses_actor, rewards, losses_critic = [], [], []
    # save csv files with losses and rewards
    metrics_dir = Path(getattr(config, "metrics_dir", Path(config.output_path) / "metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)  
    # Allow user-defined file paths on config; otherwise default names under metrics_dir
    summary_path = Path(getattr(config, "summary_path", metrics_dir / f"summary_every_{log_f}_{run_id}.csv"))
    if not summary_path.exists():
        with summary_path.open("w", newline="") as f:
            w = csv.writer(f)
            #w.writerow(["timestamp", "batch_idx", f"mean_reward_{log_f}", f"mean_actor_loss_{log_f}", f"mean_critic_loss_{log_f}", "train_cost_batch", "val_cost_batch", "cost_gap"])
            w.writerow(["timestamp", "batch_idx", f"mean_reward_{log_f}", f"mean_actor_loss_{log_f}", f"mean_critic_loss_{log_f}", "train_cost_batch", "val_cost_batch"])

    incumbent_costs = np.inf
    start_time = datetime.datetime.now()
    #logging.info(f"Time to set up before training: {start_time - t0}")
    logging.info("Starting training...")
    #t0 = start_time 
    if config.wandb:
        wandb.watch(actor, log='all', log_freq=log_f)
        wandb.log({
            'single_instance/cost0': float(training_set[0].get_costs(config.round_distances)),
            'single_instance/cost1': float(training_set[1].get_costs(config.round_distances)),
            'single_instance/cost2': float(training_set[2].get_costs(config.round_distances)),
            'single_instance/cost3': float(training_set[3].get_costs(config.round_distances)),
            'single_instance/cost4': float(training_set[4].get_costs(config.round_distances)),
        })
    #t1 = datetime.datetime.now()
    #logging.info(f"Time to plot single instance initial costs: {t1-t0}")
    #t0 = t1
    
    for batch_idx in trange(1, config.nb_train_batches + 1):
    #for batch_idx in range(1, config.nb_train_batches + 1):
        # Get a batch of training instances from the training set. Training instances are generated in advance, because
        # generating them is expensive.
        #t0 = datetime.datetime.now()
        training_set_batch_idx = batch_idx % config.nb_batches_training_set
        tr_instances = [deepcopy(instance) for instance in
                        training_set[training_set_batch_idx * batch_size: (training_set_batch_idx + 1) * batch_size]]
        #t1 = datetime.datetime.now()
        #logging.info(f"Time to copy instances: {t1-t0}")
        #t0 = t1

        # Destroy and repair the set of instances
        #if config.nb_batches_training_set - 1 <= batch_idx <= config.nb_batches_training_set + 2:
        #    destroy_instances(rng, tr_instances, config.lns_destruction, config.lns_destruction_p, debug=True)
        #else:
        #    destroy_instances(rng, tr_instances, config.lns_destruction, config.lns_destruction_p, debug=False)
        destroy_instances(rng, tr_instances, config.lns_destruction, config.lns_destruction_p)
        costs_destroyed = [instance.get_costs_incomplete(config.round_distances) for instance in tr_instances]
        avg_num_custs_destroyed = np.mean([instance.get_n_planned_customers() for instance in tr_instances])
        avg_tour_length_destroyed = np.mean(costs_destroyed)/avg_num_custs_destroyed
        if config.problem_type == 'mdvrp':
            avg_ndm_destroyed = np.mean([instance.compute_ndm(perc=True) for instance in tr_instances])
        
        # debug on instance 0
        if training_set_batch_idx == 0 and config.wandb:
            wandb.log({
                'single_instance/cost0': float(costs_destroyed[0]),
                'single_instance/cost1': float(costs_destroyed[1]),
                'single_instance/cost2': float(costs_destroyed[2]),
                'single_instance/cost3': float(costs_destroyed[3]),
                'single_instance/cost4': float(costs_destroyed[4]),
            })

        train_batch_destoyed_n_unassigned = [inst.nb_customers - inst.get_n_planned_customers() for inst in tr_instances]

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to destroy instances + stats + log single instances: {t1-t0}")
        #t0 = t1

        tour_indices, tour_logp, critic_est, policy_entropy = repair.repair(tr_instances, actor, config, critic, rng)
        costs_repaired = [instance.get_costs(config.round_distances) for instance in tr_instances]
        avg_num_custs_repaired = np.mean([instance.get_n_planned_customers() for instance in tr_instances])
        avg_tour_length_repaired = np.mean(costs_repaired)/avg_num_custs_repaired
        if config.problem_type == 'mdvrp':
            avg_ndm_repaired = np.mean([instance.compute_ndm(perc=True) for instance in tr_instances])

        # debug on instance 0
        if training_set_batch_idx == 0 and config.wandb:
            wandb.log({
                'single_instance/cost0': float(costs_repaired[0]),
                'single_instance/cost1': float(costs_repaired[1]),
                'single_instance/cost2': float(costs_repaired[2]),
                'single_instance/cost3': float(costs_repaired[3]),
                'single_instance/cost4': float(costs_repaired[4]),
            })
         
        unscaled_costs_repaired = deepcopy(costs_repaired)

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to repair instances + stats + log single instances: {t1-t0}")
        #t0 = t1
            #scale costs
        #if config.scale_rewards:
        #    batch_init_costs = [deepcopy(cost) for cost in init_costs[training_set_batch_idx*batch_size: (training_set_batch_idx +1)*batch_size]]
        #    costs_repaired = np.array(costs_repaired) / np.array(batch_init_costs)
        #    costs_destroyed = np.array(costs_destroyed) / np.array(batch_init_costs) 

        # Reward/Advantage computation
        reward = np.array(costs_repaired) - np.array(costs_destroyed) #per instance

        if config.scale_rewards:
            reward = reward / np.array(costs_destroyed)

        reward = torch.from_numpy(reward).float().to(config.device)
        advantage = reward - critic_est # per instance

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to compute rewards and advantage: {t1-t0}")
        #t0 = t1

        # Actor loss computation and backpropagation
        actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1)) # mean over batch
        actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_total_grad_preclip = torch.nn.utils.get_total_norm(actor.parameters())
        torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
        actor_total_grad_postclip = torch.nn.utils.get_total_norm(actor.parameters())

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to compute actor loss, compute tensor gradient and clip actor gradient: {t1-t0}")
        #t0 = t1

        actor_gs = grad_stats(actor)

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to compute actor gradient stats: {t1-t0}")
        #t0 = t1

        actor_optim.step()

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to do actor optim step: {t1-t0}")
        #t0 = t1

        # Critic loss computation and backpropagation
        critic_loss = torch.mean(advantage ** 2)
        critic_optim.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_total_grad_preclip = torch.nn.utils.get_total_norm(critic.parameters())
        torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
        critic_total_grad_postclip = torch.nn.utils.get_total_norm(critic.parameters())

        ##t1 = datetime.datetime.now()
        ##logging.info(f"Time to compute critic loss, compute tensor gradient and clip critic gradient: {t1-t0}")
        ##t0 = t1

        critic_gs = grad_stats(critic)

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to compute critic gradient stats: {t1-t0}")
        #t0 = t1

        critic_optim.step()

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to do critic optim step: {t1-t0}")
        #t0 = t1

        rewards.append(torch.mean(reward.detach()).item()) # appending mean rewards of batch
        losses_actor.append(torch.mean(actor_loss.detach()).item())
        losses_critic.append(torch.mean(critic_loss.detach()).item())

        #t1 = datetime.datetime.now()
        #logging.info(f"Time to append rewards and losses: {t1-t0}")
        #t0 = t1

        # Replace the solution of the training set instances with the new created solutions
        for i in range(batch_size):
            training_set[training_set_batch_idx * batch_size + i] = tr_instances[i]
        
        #t1 = datetime.datetime.now()
        #logging.info(f"Time to replace training_set instances: {t1-t0}")
        #t0 = t1

        # Log performance every log_f batches
        if batch_idx % log_f == 0 and batch_idx > 0:
            #t0 = datetime.datetime.now()
            mean_loss = np.mean(losses_actor[-log_f:])  #avg actor loss over the last log_f batches
            mean_critic_loss = np.mean(losses_critic[-log_f:]) #avg critic loss over the last log_f batches
            mean_reward = np.mean(rewards[-log_f:]) #avg reward of last log_f batches
            # cost of this batch (multiple of log_f)
            train_cost_batch = np.mean(unscaled_costs_repaired) # mean repair cost OF THE CURRENT BATCH
            train_cost_batch_destroyed = np.mean(costs_destroyed) 

            #t1 = datetime.datetime.now()
            #logging.info(f"Time to compute stats to log: {t1-t0}")
            #t0 = t1

            val_cost_snapshot = lns_validation_search(validation_instances, actor, config, rng) # mean lns cost over validation_instances
            #cost_gap = (train_cost_batch - val_cost_snapshot) / val_cost_snapshot if val_cost_snapshot != 0 else 0.0

            #t1 = datetime.datetime.now()
            #logging.info(f"Time to do val_cost_snapshot: {t1-t0}")
            #t0 = t1

            logging.info(
                f'Batch {batch_idx}/{config.nb_train_batches}, repair costs (reward): {mean_reward:2.3f}, loss: {mean_loss:2.6f} , critic_loss: {mean_critic_loss:2.6f}')
            
            now = datetime.datetime.now().isoformat()
            with summary_path.open("a", newline="") as f:
                w = csv.writer(f)
                #w.writerow([now, batch_idx, mean_reward, mean_loss, mean_critic_loss, train_cost_batch, val_cost_snapshot, cost_gap])
                w.writerow([now, batch_idx, mean_reward, mean_loss, mean_critic_loss, train_cost_batch, val_cost_snapshot])

            #t1 = datetime.datetime.now()
            #logging.info(f"Time write to csv file: {t1-t0}")
            #t0 = t1

            if config.wandb:
                wandb.log({
                    'batch_idx': int(batch_idx), 
                    'train/reward': float(mean_reward), 
                    'train/actor_loss': float(mean_loss), 
                    'train/critic_loss': float(mean_critic_loss),
                    'train/train_cost_batch': float(train_cost_batch),
                    'train/train_cost_batch_destroyed': float(train_cost_batch_destroyed),
                    'train/destroy_batch_avg_unassigned': float(sum(train_batch_destoyed_n_unassigned)/len(train_batch_destoyed_n_unassigned)),
                    'train/val_cost_snapshot': float(val_cost_snapshot),
                    'train/min_costs_destroyed_batch': float(min(costs_destroyed)),
                    'train/max_costs_destroyed_batch': float(max(costs_destroyed)),
                    'train/policy_entropy_mean': float(np.mean(policy_entropy.cpu().detach().numpy())),
                    'train/policy_entropy_var': float(np.var(policy_entropy.cpu().detach().numpy())),
                    'tour_stats/average_tour_length_destroyed': float(avg_tour_length_repaired),
                    'tour_stats/average_tour_length_repaired': float(avg_tour_length_repaired),
                    'adv/mean': float(advantage.mean()),
                    'adv/std': float(advantage.std()),
                    'adv/abs_mean': float(advantage.abs().mean()),
                    
                    'grads/actor/total_preclip': float(actor_total_grad_preclip),
                    'grads/actor/total_postclip': float(actor_total_grad_postclip),
                    'grads/critic/total_preclip': float(critic_total_grad_preclip),
                    'grads/critic/total_postclip': float(critic_total_grad_postclip),
                    'grads/max_grad_norm': float(config.max_grad_norm),

                    'grads/actor/mean': actor_gs['agg']['grad_norm/mean'],
                    'grads/actor/zero_params': actor_gs['zero_grad_params'],
                    'grads/actor/nan_inf': actor_gs['nan_inf_grads'],

                    'grads/critic/mean': critic_gs['agg']['grad_norm/mean'],
                    'grads/critic/zero_params': critic_gs['zero_grad_params'],
                    'grads/critic/nan_inf': critic_gs['nan_inf_grads'],

                    'weight_norm/actor/mean': actor_gs['agg']['weight_norm/mean'],
                    'weight_norm/critic/mean': critic_gs['agg']['weight_norm/mean'],
                })
                if config.problem_type == 'mdvrp':
                    wandb.log({
                        'tour_stats/avg_ndm_destroyed ': float(avg_ndm_destroyed),
                        'tour_stats/avg_ndm_repaired ': float(avg_ndm_repaired),
                    })
                      
            #t1 = datetime.datetime.now()
            #logging.info(f"Time to log everything on wandb: {t1-t0}")
            #t0 = t1

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
    if config.wandb:
        # end wandb logging
        wandb.finish()
    return incumbent_model_path


def lns_validation_search(validation_instances, actor, config, rng):
    validation_instances_copies = [deepcopy(instance) for instance in validation_instances]
    actor.eval()
    operation = LnsOperatorPair(actor, config.lns_destruction, config.lns_destruction_p)
    costs, _ = lns_batch_search(validation_instances_copies, config.lns_max_iterations,
                                config.lns_timelimit_validation, [operation], config, rng)
    actor.train()
    return np.mean(costs)

def grad_stats(module):
    total_params = 0
    zero_grad_params = 0
    nan_inf_grads = 0
    layer_stats = []  # (name, grad_norm, weight_norm)

    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        total_params += 1
        g = p.grad
        if g is None:
            zero_grad_params += 1
            continue
        if torch.isnan(g).any() or torch.isinf(g).any():
            nan_inf_grads += 1
        gn = g.norm(2).item()
        wn = p.data.norm(2).item()
        layer_stats.append((name, gn, wn))
    if layer_stats:
        grad_norm = np.array([x[1] for x in layer_stats], dtype=float)
        weight_norm = np.array([x[2] for x in layer_stats], dtype=float)
        agg = {
            "grad_norm/mean": float(grad_norm.mean()),
            "weight_norm/mean": float(weight_norm.mean()),
        }
    else:
        agg = {"grad_norm/mean": 0.0, "weight_norm/mean": 0.0}

    return {
        "total_params": total_params,
        "zero_grad_params": zero_grad_params,
        "nan_inf_grads": nan_inf_grads,
        "layer_stats": layer_stats,
        "agg": agg
        }

