import copy 
import torch 
import torch.nn as nn 
import numpy as np
from utils import load_model, load_data, setup_cfg, read_config, seed_everything
from graphmae.models import build_model
class Client_struct:
    def __init__(self,client_id,args):
        self.client_id = client_id
        ## dataload
        config = {}
        config['client_id'] = client_id
        config['train_all'] = True
        config['batch_size'] = 2048
        config['self_loop'] = True
        config['add_attr'] = True
        #只载入自身拥有的数据
        self.dataloader,_ = load_data(config)
        ## model
        self.model : nn.Module = build_model(args)
        self.lr = args.lr
        self.device = args.device if args.device >= 0 else "cpu"

    def load_model(self,model_param):
        self.model.load_state_dict(model_param)
    
    def save_param(self):
        return copy.deepcopy(self.model.state_dict())
    # 使用本地数据进行训练
    def train_epoch(self):
        optimizer = torch.optim.SGD(self.model.parameters(),lr = self.lr)
        self.model.to(self.device)
        self.model.train()
        loss_list = []
        for batch_g,_ in self.dataloader:
            batch_g = batch_g.to(self.device)
            feat = batch_g.ndata["attr"]
            loss, loss_dict = self.model(batch_g, feat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        self.model.to('cpu')
        return np.mean(loss_list)
    
