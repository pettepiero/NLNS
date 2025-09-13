# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py
from tqdm import tqdm
import torch
import numpy as np
from vrp import vrp_problem
from vrp import mdvrp_problem
import torch.nn.functional as F


def get_depot_mask(batch_size, n_points, config, instances):
    
    # depot mask per batch. Per instance, the first instance.n_depots indices are depots
    depot_mask = torch.zeros((batch_size, n_points), dtype=torch.bool, device=config.device)
    for i, inst in enumerate(instances):
        depot_mask[i, :inst.n_depots] = True

    return depot_mask

def _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng):
    batch_size, n_points, _ = static_input.shape

    depot_mask = get_depot_mask(batch_size, n_points, config, instances)

    tour_idx, tour_logp = [], []

    instance_repaired = np.zeros(batch_size)

    origin_idx = np.zeros((batch_size), dtype=int)
    iter = 0
    while not instance_repaired.all():
        iter += 1
        # if origin_idx == 0 select the next tour end that serves as the origin at random
        print(f"DEBUG: iter {iter} in _actor_model_forward")
        for i, instance in enumerate(tqdm(instances)):
            if (origin_idx[i] == 0 or int(origin_idx[i]) in instance.depot_indices) and not instance_repaired[i]:
                if rng is None:
                    origin_idx[i] = np.random.choice(instance.open_nn_input_idx, 1).item()
                else:
                    origin_idx[i] = rng.choice(instance.open_nn_input_idx, 1).item()

        if config.problem_type == 'mdvrp':
            mask = mdvrp_problem.get_mask(origin_idx, dynamic_input, instances, config, vehicle_capacity).to(config.device)
            print("DEBUG: got mask")
        elif config.problem_type == 'vrp':
            mask = vrp_problem.get_mask(origin_idx, dynamic_input, instances, config, vehicle_capacity).to(config.device).float()
        else:
            raise ValueError(
                f"Problem in config.problem_type: expected either 'mdvrp' or 'vrp', got {config.problem_type}."
            ) 
            
        # Rescale customer demand based on vehicle capacity
        dynamic_input_float = dynamic_input.float()
        dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(vehicle_capacity)

        origin_static_input = static_input[torch.arange(batch_size), origin_idx]
        origin_dynamic_input_float = dynamic_input_float[torch.arange(batch_size), origin_idx]

        print(f"DEBUG: calling actor.forward")
        # Forward pass. Returns a probability distribution over the point (tour end or depot) that origin should be connected to
        probs = actor.forward(
                static_input                = static_input, 
                dynamic_input_float         = dynamic_input_float,
                origin_static_input         = origin_static_input, 
                origin_dynamic_input_float  = origin_dynamic_input_float, 
                mask                        = mask,
                depot_mask                  = depot_mask
                )
        print(f"DEBUG: actor.forward is done")
        #masked_logits = logits.masked_fill(~mask, float('-inf'))
        #probs = F.softmax(masked_logits, dim=1)
        #probs = F.softmax(probs + mask.log(), dim=1)  # Set prob of masked tour ends to zero


        if actor.training:
            print(f"DEBUG: in actor.training")
            m = torch.distributions.Categorical(probs=probs)
            #m = torch.distributions.Categorical(logits=masked_logits)
            print(f"DEBUG: sampled m")
            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            ptr = m.sample()
            #print(f"DEBUG: type(masked_logist): {type(masked_logits)}")
            #print(f"DEBUG: ptr : {ptr}")
            print(f"Entering while loop")
            valid = mask.gather(1, ptr.unsqueeze(1)).squeeze(1)
            tries = 0
            while (~valid).any():
                new_ptr = m.sample()
                ptr = torch.where(valid, ptr, new_ptr)
                valid = mask.gather(1, ptr_squeeze(1)).squeeze(1)
                tries += 1
                if tries > 1000:
                    raise RuntimeError("Could not sample a valid action")
            #while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
            #    ptr = m.sample()
            print(f"Out of while")
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs, 1)  # Greedy selection
            logp = prob.log()

        # Perform action  for all instances sequentially
        nn_input_updates = []
        ptr_np = ptr.cpu().numpy()
        print(f"DEBUG: performing actions")
        for i, instance in enumerate(tqdm(instances)):
            idx_from = origin_idx[i].item()
            idx_to = ptr_np[i]
            if idx_from in instance.depot_indices and idx_to in instance.depot_indices:  # No need to update in this case
                continue
            nn_input_update, cur_nn_input_idx = instance.do_action(idx_from, idx_to)  # Connect origin to select point
            for s in nn_input_update:
                s.insert(0, i)
                nn_input_updates.append(s)

            # Update origin
            if len(instance.open_nn_input_idx) == 0:
                instance_repaired[i] = 1
                origin_idx[i] = 0  # If instance is repaired set origin to 0
            else:
                origin_idx[i] = cur_nn_input_idx  # Otherwise, set to tour end of the connect tour

        # Update network input
        if len(nn_input_updates) > 0:
            nn_input_update = np.array(nn_input_updates, dtype=np.long)
            nn_input_update = torch.from_numpy(nn_input_update).to(config.device).long()
            dynamic_input[nn_input_update[:, 0], nn_input_update[:, 1]] = nn_input_update[:, 2:]

        logp = logp * (1. - torch.from_numpy(instance_repaired).float().to(config.device))
        tour_logp.append(logp.unsqueeze(1))
        tour_idx.append(ptr.data.unsqueeze(1))

    tour_idx = torch.cat(tour_idx, dim=1)
    tour_logp = torch.cat(tour_logp, dim=1)
    return tour_idx, tour_logp


def _critic_model_forward(critic, static_input, dynamic_input, batch_capacity):
    dynamic_input_float = dynamic_input.float()

    dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(batch_capacity)

    return critic.forward(static_input, dynamic_input_float).view(-1)


def repair(instances, actor, config, critic=None, rng=None):
    nb_input_points = max([instance.get_max_nb_input_points() for instance in instances])  # Max. input points of batch
    batch_size = len(instances)

    # Create batch input
    static_input = np.zeros((batch_size, nb_input_points, 2))
    dynamic_input = np.zeros((batch_size, nb_input_points, 2), dtype='int')
    print(f"DEBUG: looping over instances in repair:")
    for i, instance in enumerate(tqdm(instances)):
        static_nn_input, dynamic_nn_input = instance.get_network_input(nb_input_points)
        static_input[i] = static_nn_input
        dynamic_input[i] = dynamic_nn_input

    print(f"DEBUG: finished loop in repair")
    static_input = torch.from_numpy(static_input).to(config.device).float()
    dynamic_input = torch.from_numpy(dynamic_input).to(config.device).long()
    vehicle_capacity = instances[0].capacity # Assumes that the vehicle capcity is identical for all instances of the batch

    cost_estimate = None
    if critic is not None:
        cost_estimate = _critic_model_forward(critic, static_input, dynamic_input, vehicle_capacity)
    print(f"DEBUG: calling _actor_model_forward")
    tour_idx, tour_logp = _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng)

    return tour_idx, tour_logp, cost_estimate

