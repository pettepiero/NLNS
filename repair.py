# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py
from tqdm import tqdm
import torch
import numpy as np
from vrp import vrp_problem
from vrp import mdvrp_problem
import torch.nn.functional as F
from copy import deepcopy
import argparse
import logging

logger = logging.getLogger(__name__)

def get_depot_mask(batch_size: int, n_points: int, config: argparse.Namespace, instances: list) -> torch.Tensor:
    """
    Returns a boolean tensor with True in positions of depots of the batch.

    :param batch_size: Batch size
    :type batch_size: int
    :param n_points: Integer representing the max number of open ends in batch
    :type n_points: int
    :param config: Namespace of config options
    :type config: argparse.Namespace
    :param instances: List of instances in the batch
    :type instances: list
    :raises AssertionError: When len(instances) != batch_size
    :return: Tensor with True in positions of depots
    :rtype: torch.Tensor
    """
    # depot mask per batch. Per instance, the first instance.n_depots indices are depots
    assert len(instances) == batch_size
    depot_mask = torch.zeros((batch_size, n_points), dtype=torch.bool, device=config.device)
    for i, inst in enumerate(instances):
        depot_mask[i, :inst.n_depots] = True

    return depot_mask

def _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng):
    """
    
    """
    batch_size, n_points, _ = static_input.shape
    # get depot_mask
    depot_mask = get_depot_mask(batch_size, n_points, config, instances)
    # set up variables
    tour_idx, tour_logp = [], []
    instance_repaired = np.zeros(batch_size)
    origin_idx = np.zeros((batch_size), dtype=int)
    policy_entropy = None

    iter = 0
    while not instance_repaired.all():
        iter += 1
        # if origin_idx == 0 select the next tour end that serves as the origin at random
        for i, instance in enumerate(instances):
            if (origin_idx[i] == 0 or int(origin_idx[i]) in instance.depot_indices) and not instance_repaired[i]:
                if rng is None:
                    origin_idx[i] = np.random.choice(instance.open_nn_input_idx, 1).item()
                else:
                    origin_idx[i] = rng.choice(instance.open_nn_input_idx, 1).item()
        # get mask
        if config.problem_type == 'mdvrp':
            mask = mdvrp_problem.get_mask(origin_idx, dynamic_input, instances, config, vehicle_capacity).to(config.device)
        elif config.problem_type == 'vrp':
            #mask = vrp_problem.get_mask(origin_idx, dynamic_input, instances, config, vehicle_capacity).to(config.device).float()
            mask = vrp_problem.get_mask(origin_idx, dynamic_input, instances, config, vehicle_capacity).to(config.device)
        else:
            raise ValueError(
                f"Problem in config.problem_type: expected either 'mdvrp' or 'vrp', got {config.problem_type}."
            ) 
        # Rescale customer demand based on vehicle capacity
        dynamic_input_float = dynamic_input.float()
        dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0] / float(vehicle_capacity)
        origin_static_input = static_input[torch.arange(batch_size), origin_idx]
        origin_dynamic_input_float = dynamic_input_float[torch.arange(batch_size), origin_idx]

        # Forward pass. Returns a probability distribution over the point (tour end or depot) that origin should be connected to
        probs, unmasked_probs = actor.forward(
                static_input                = static_input, 
                dynamic_input_float         = dynamic_input_float,
                origin_static_input         = origin_static_input, 
                origin_dynamic_input_float  = origin_dynamic_input_float, 
                mask                        = mask,
                depot_mask                  = depot_mask
                )
        unmasked_m = torch.distributions.Categorical(probs=unmasked_probs)
        policy_entropy = unmasked_m.entropy()

        if actor.training:
            # sample actions
            m = torch.distributions.Categorical(probs=probs)
            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            ptr = m.sample()
            logp = m.log_prob(ptr)
        else:
            # greedy actions
            prob, ptr = torch.max(probs, 1)
            logp = prob.log()

        # Perform action  for all instances sequentially
        nn_input_updates = []
        ptr_np = ptr.cpu().numpy()
        for i, instance in enumerate(instances):
            idx_from = origin_idx[i].item()
            idx_to = ptr_np[i].item()
            #assert not (idx_from in instance.depot_indices and idx_to in instance.depot_indices), "Trying to connect depot to depot"
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

    # END OF WHILE HERE
    tour_idx = torch.cat(tour_idx, dim=1)
    tour_logp = torch.cat(tour_logp, dim=1)
    return tour_idx, tour_logp, policy_entropy


def _expected_dyn_for_idx(inst, dyn_input, row_i, col_j, n_depots):
    # state lives in [:, :, 1], demand in [:, :, 0]
    is_depot_col = (col_j < n_depots)
    dyn_dem = dyn_input[row_i, col_j, 0].item()
    dyn_state = dyn_input[row_i, col_j, 1].item()

    tour, pos = inst.nn_input_idx_to_tour[col_j]
    # "True" tour load (sum of customer demands in that tour)
    true_load = sum(l[1] for l in tour)

    if is_depot_col:
        # Depot rows are encoded as (-capacity, state=-1) regardless of tour load.
        assert dyn_state == -1, f"Depot col {col_j} must have state -1, got {dyn_state}"
        assert dyn_dem == -inst.capacity, \
            f"Depot col {col_j} must have demand -capacity ({-inst.capacity}), got {dyn_dem}"
        # For depots, we DON'T compare dyn_dem to true_load.
        return None, None  # signal: skip equality check for depots
    else:
        # Non-depot NN inputs (first/last of incomplete tours or single-customer tours)
        # must carry the tour's current load in the tensor.
        assert dyn_state in (1, 2, 3), f"Unexpected state {dyn_state} at col {col_j}"
        return true_load, dyn_dem



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
    for i, instance in enumerate(instances):
        static_nn_input, dynamic_nn_input = instance.get_network_input(nb_input_points)
        static_input[i] = static_nn_input
        dynamic_input[i] = dynamic_nn_input

    static_input = torch.from_numpy(static_input).to(config.device).float()
    dynamic_input = torch.from_numpy(dynamic_input).to(config.device).long()
    assert all(inst.capacity == instances[0].capacity for inst in instances), \
            "All instances in the batch must share the same capacity."
    vehicle_capacity = instances[0].capacity # Assumes that the vehicle capcity is identical for all instances of the batch

    cost_estimate = None
    if critic is not None:
        cost_estimate = _critic_model_forward(critic, static_input, dynamic_input, vehicle_capacity)
    tour_idx, tour_logp, policy_entropy = _actor_model_forward(actor, instances, static_input, dynamic_input, config, vehicle_capacity, rng)

    return tour_idx, tour_logp, cost_estimate, policy_entropy

