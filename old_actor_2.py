# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.embed_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        output = F.relu(self.embed(input))
        output = self.embed_2(output)
        return output

class Attention(nn.Module):
    def __init__(self, device, hidden_size):
        super(Attention, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, hidden), 1)

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


class Pointer(nn.Module):

    def __init__(self, device, hidden_size):
        super(Pointer, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.encoder_attn = Attention(device, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, all_hidden, origin_hidden):
        enc_attn = self.encoder_attn(all_hidden, origin_hidden.transpose(2, 1).squeeze(1))
        context = enc_attn.bmm(all_hidden.permute(0, 2, 1))

        #input = torch.cat((origin_hidden.squeeze(), context.squeeze()), dim=1)
        assert origin_hidden.dim() >= 2 and context.dim() >= 2, "Pointer expects batched tensors"
        input = torch.cat((origin_hidden.flatten(1), context.flatten(1)), dim=1)  # (B, 2H)

        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))

        output = output.unsqueeze(2)
        output = output.expand_as(all_hidden)

        v = self.v.expand(all_hidden.size(0), -1, -1)
        probs = torch.bmm(v, torch.tanh(all_hidden + output)).squeeze(1)

        return probs


class VrpActorModel(nn.Module):
    def __init__(self, device, hidden_size=128):
        super(VrpActorModel, self).__init__()
        max_depots = 10

        #self.all_embed      = Encoder(4, hidden_size)
        self.first_encoder  = Encoder(4, 64) 
        self.depot_table    = nn.Embedding(max_depots+1, 64)
        self.origin_embed   = Encoder(4, 64)
        
        self.pointer        = Pointer(device, hidden_size)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static_input, dynamic_input_float, origin_static_input, origin_dynamic_input_float, mask, depot_mask):
        # Set the input feature values of already visited customers (demand == 0) to zero
        # customers are all but depots
        is_customer = ~depot_mask
        is_active = dynamic_input_float[:, :, 1] >= 0
        keep = depot_mask | (is_customer & is_active)
        keep3 = keep.unsqueeze(2).float()
        static_input = static_input * keep3
        n_depots = sum(depot_mask[0]).item() # all batch is of same blueprint

        node_features = torch.cat((static_input, dynamic_input_float), dim=2)
        cust_features = node_features[:,n_depots:,:] 
        depot_features = node_features[:,:n_depots,:] 

        all_embeddings = self.first_encoder(node_features)

        depot_ids = node_features[:,:,-1].long() # [B, N]
        depot_ids[depot_ids < 0] = -depot_ids[depot_ids < 0]   # -1→1, -2→2
        cust_depot_emb = self.depot_table(depot_ids)

        final_embeddings = torch.cat((all_embeddings, cust_depot_emb), dim=-1)

        origin_hidden = self.origin_embed.forward(
            torch.cat((origin_static_input.unsqueeze(1), origin_dynamic_input_float.unsqueeze(1)), dim=2))
        origin_depot_ids = origin_dynamic_input_float[:, -1].long()
        origin_depot_ids[origin_depot_ids < 0] = -origin_depot_ids[origin_depot_ids < 0]
        origin_depot_embed = self.depot_table(origin_depot_ids).unsqueeze(1)
        final_origin_embedding = torch.cat((origin_hidden, origin_depot_embed), dim=-1)

        logits = self.pointer.forward(final_embeddings.permute(0, 2, 1), final_origin_embedding.permute(0, 2, 1))

        #masked_logits = logits.masked_fill(~mask, -1e9)
        masked_logits = torch.where(mask, logits, torch.full_like(logits, float('-inf')))
        probs = F.softmax(masked_logits, dim=1)
        unmasked_probs = F.softmax(logits, dim=1)
        return probs, unmasked_probs
