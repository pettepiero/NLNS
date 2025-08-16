import torch
import torch.nn as nn
from vrp.data_utils import read_instance
import time
import math
import search
import numpy as np
from vrp.mdvrp_problem import get_mask

class DummyActorModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=16):
        super(DummyActorModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

        # random normal initialization
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
        # bias to zero
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, static, dynamic, origin_static, origin_dynamic, mask):
        x = torch.cat([static, dynamic], dim=2)  # [B, N, 4]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)  # [B, N]
        return x

def dummy_model(instance_path, config, actor=None):
    instance = read_instance(instance_path)
    start_time = time.time()
    instance.create_initial_solution()
    incumbent_costs = instance.get_costs(config.round_distances)
    instance.verify_solution(config)

    if actor is None:
        actor = DummyActorModel().to(config.device)
    
    # Step 1: destroy using point-based destruction
    destroy_procedure = "P"
    p_destruction = 0.3  # hard coded probability
    search.destroy_instances([instance], destroy_procedure, p_destruction)

    nb_input_points = instance.get_max_nb_input_points()
    static_input, dynamic_input = instance.get_network_input(nb_input_points)
    
    # Step 2: simple repair loop using actor model
    while len(instance.open_nn_input_idx) > 1:
        # Build input tensors
        nb_input_points = instance.get_max_nb_input_points()
        static_input_np, dynamic_input_np = instance.get_network_input(nb_input_points)

        static_input = torch.from_numpy(static_input_np).unsqueeze(0).float().to(config.device)    # [1, N, 2]
        dynamic_input = torch.from_numpy(dynamic_input_np).unsqueeze(0).float().to(config.device)  # [1, N, 2]

        # Pick random origin from open tour ends
        idx_from = np.random.choice(instance.open_nn_input_idx)
        origin_static = static_input[0, idx_from].unsqueeze(0)     # [1, 2]
        origin_dynamic = dynamic_input[0, idx_from].unsqueeze(0)   # [1, 2]

        # Create mask
        origin_idx_tensor = torch.tensor([idx_from], device=config.device)
        mask = get_mask(origin_idx_tensor, dynamic_input.long(), [instance], config, instance.capacity).float()

        # Forward pass through the dummy actor
        with torch.no_grad():
            logits = actor(static_input, dynamic_input, origin_static, origin_dynamic, mask)  # [1, N]
            probs = torch.softmax(logits + mask.log(), dim=1)  # apply mask via log-trick

        # Sample or choose best
        idx_to = torch.argmax(probs, dim=1).item()

        # Perform action
        nn_input_update, curr_input_idx = instance.do_action(idx_from, idx_to)
        
    # Step 3: Fix depot start/end if needed
    for tour in instance.solution:
        if tour[0][0] not in instance.depot_indices:
            if tour[-1][0] not in instance.depot_indices:
                depot = np.random.choice(instance.depot_indices)
                tour.insert(0, [depot, 0, None])
                tour.append([depot, 0, None])
            else:
                depot = tour[-1][0]
                tour.insert(0, [depot, 0, None])
        elif tour[0][0] == tour[-1][0]:
            continue
        else:
            depot = tour[0][0]
            tour.append([depot, 0, None])

    cost = instance.get_costs(config.round_distances)
    duration = time.time() - start_time
    print(f"Instance.solution:\n")
    for el in instance.solution:
        print(el)
    return cost, duration

