import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set

from dgllife.model import GIN 
import sys
sys.path.append('.')

class GINP(nn.Module):

    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5,
                 emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1):
        super(GINP, self).__init__()

        if num_layers < 2:
            raise ValueError('Number of GNN layers must be greater '
                             'than 1, got {:d}'.format(num_layers))

        self.gnn = GIN(num_node_emb_list=num_node_emb_list,
                       num_edge_emb_list=num_edge_emb_list,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       JK=JK,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            if JK == 'concat':
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear((num_layers + 1) * emb_dim, 1))
            else:
                self.readout = GlobalAttentionPooling(
                    gate_nn=nn.Linear(emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set()
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', "
                             "'max', 'attention' or 'set2set', got {}".format(readout))

        if JK == 'concat':
            self.predict = nn.Linear((num_layers + 1) * emb_dim, n_tasks)
            self.w = nn.Linear(256,(num_layers + 1) * emb_dim)
        else:
            self.predict = nn.Linear(emb_dim, n_tasks)
            self.w = nn.Linear(256,emb_dim)
        self.smodel = torch.load('save/s_model/s_model.pkl')
        

    def forward(self, g, categorical_node_feats, categorical_edge_feats):
        node_feats = self.gnn(g, categorical_node_feats, categorical_edge_feats)
        n_f = self.w(self.smodel.embed(g,g.ndata['attr']).detach())
        node_feats = node_feats + n_f
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

class GINE_S(nn.Module):
    def __init__(self, num_node_emb_list, num_edge_emb_list, num_layers=5, emb_dim=300, JK='last', dropout=0.5, readout='mean', n_tasks=1) -> None:
        super().__init__()
        self.model_predictor = GINP(num_node_emb_list, num_edge_emb_list, num_layers=num_layers, emb_dim=emb_dim, JK=JK, dropout=dropout, readout=readout, n_tasks=n_tasks)
        
    def forward(self, g, x, w):
        x = x.long().split(1, dim=-1)
        w = w.long().split(1, dim=-1)
        
        x = [_.squeeze() for _ in x]
        w = [_.squeeze() for _ in w]
        return self.model_predictor(g, x, w)
        