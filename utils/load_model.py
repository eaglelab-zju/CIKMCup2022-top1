import torch.nn as nn

def load_model(cfg, dataset):
    if cfg['model'] == 'GINE':
        # dgl lifesci
        from models.GINE import GINE
        model = GINE(
            num_node_emb_list=dataset.num_node_emb_list,
            num_edge_emb_list=dataset.num_edge_emb_list,
            num_layers=cfg['num_layers'],
            emb_dim=cfg['emb_dim'],
            JK=cfg['jk'],
            dropout=cfg['drop_ratio'],
            readout=cfg['readout'],
            n_tasks=dataset.num_tasks
        )
    elif cfg['model'] == 'GIN':
        # dgl example
        from models.GIN import GIN
        model = GIN(
            input_dim=dataset.nfeat_dim,
            hidden_dim=cfg['emb_dim'],
            output_dim=dataset.num_tasks
        )
    elif cfg['model'] == 'PAGTN':
        # dgl lifesci
        from dgllife.model import PAGTNPredictor
        model = PAGTNPredictor(
            node_in_feats=dataset.nfeat_dim, 
            node_out_feats=cfg['emb_dim'], 
            node_hid_feats=cfg['emb_dim'], 
            edge_feats=dataset.efeat_dim, 
            depth=cfg['num_layers'], 
            nheads=cfg['num_heads'], 
            dropout=cfg['drop_ratio'], 
            activation=nn.LeakyReLU(negative_slope=0.2), 
            n_tasks=dataset.num_tasks, 
            mode=cfg['readout']
        )
    elif cfg['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(
            node_in_feats=dataset.nfeat_dim, 
            edge_in_feats=dataset.efeat_dim, 
            node_out_feats=cfg['emb_dim'], 
            edge_hidden_feats=128, 
            n_tasks=dataset.num_tasks, 
            num_step_message_passing=6, 
            num_step_set2set=6, 
            num_layer_set2set=3
        )
    elif cfg['model'] == 'AttentiveFP':
        from dgllife.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(
            node_feat_size=dataset.nfeat_dim, 
            edge_feat_size=dataset.efeat_dim, 
            num_layers=cfg['num_layers'], 
            num_timesteps=2, 
            graph_feat_size=cfg['emb_dim'], 
            n_tasks=dataset.num_tasks, 
            dropout=cfg['drop_ratio'])
    elif cfg['model'] == 'GINE_S':
        from models.GINE_S import GINE_S
        model = GINE_S(
            num_node_emb_list=dataset.num_node_emb_list,
            num_edge_emb_list=dataset.num_edge_emb_list,
            num_layers=cfg['num_layers'],
            emb_dim=cfg['emb_dim'],
            JK=cfg['jk'],
            dropout=cfg['drop_ratio'],
            readout=cfg['readout'],
            n_tasks=dataset.num_tasks
        )
    elif cfg['model'] == 'GNN-dgl':
        from models.GNN import Net
        model = Net(
            num_node_emb_list=dataset.num_node_emb_list,
            num_edge_emb_list=dataset.num_edge_emb_list,
            num_layers=cfg['num_layers'],
            emb_dim=cfg['emb_dim'],
            JK=cfg['jk'],
            readout=cfg['readout'],
            APPNP=cfg['appnp'],
            node_degree=cfg['degree'],
            dropout=cfg['drop_ratio'],
            n_tasks=dataset.num_tasks
        )
    else:
        raise ValueError(f"{cfg['model']} type model not exist!")
        
    return model