import torch
import torch.nn as nn

from dgllife.model import GINPredictor

class GINE(nn.Module):
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5, emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1) -> None:
        super().__init__()
        self.model_predictor = GINPredictor(num_node_emb_list, num_edge_emb_list, num_layers=num_layers, emb_dim=emb_dim, JK=JK, dropout=dropout, readout=readout, n_tasks=n_tasks)
        
    def forward(self, g, x, w):
        x = x.long().split(1, dim=-1)
        w = w.long().split(1, dim=-1)
        
        x = [_.squeeze() for _ in x]
        w = [_.squeeze() for _ in w]
        return self.model_predictor(g, x, w)
        