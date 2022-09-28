import os
import copy 
import argparse
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader

import collections
import copy
from tqdm import tqdm 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import nni
from dataset import AnalytiCupDataset
from utils import load_model, load_data, load_optim, setup_cfg, read_config, seed_everything
from models.maepre import PreModel

class Client:
    def __init__(self, config:dict):
        self.config = config 
        self.task = 'cls' if config['client_id'] < 9 else 'reg'
        self.train_loader, self.val_loader = load_data(config)
        self.model = load_model(config, self.train_loader.dataset)
        self.optimizer, self.scheduler = load_optim(config, self.model)

    def model_param(self):
        return self.model.cpu().state_dict()

    def update_param(self, param):
        self.model.load_state_dict(param)

    def loss_func(self, outputs, labels):
        if self.config['loss'] == 'focal':
            focal_loss = FocalLoss(gamma=1)
            labels = labels.view(-1)
            loss = focal_loss(outputs, labels)
        elif self.task == 'cls':
            labels = labels.view(-1)
            loss = F.cross_entropy(outputs, labels)
        else:
            loss = F.mse_loss(outputs, labels)
        return loss 

    def pre_train(self,max_epoch):
        pre_model = PreModel(self.config['emb_dim'],self.config['num_layers'],self.train_loader.dataset,
                    self.config['jk'],self.config['drop_ratio'],
                    mask_rate=self.config['mask'],
                    drop_edge_rate=0,
                    replace_rate=0,
                    alpha_l=2)
        pre_model.to(self.config['device'])
        optimizer = torch.optim.Adam(pre_model.parameters(),lr=0.001)
        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
            pre_model.train()
            for batch_g, _ in self.train_loader:
                batch_g = batch_g.to(self.config['device'])
                feat = batch_g.ndata["feat"]
                weight = batch_g.edata['weight']
                loss, loss_dict = pre_model(batch_g, feat,weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        para = pre_model.enc_params
        param = collections.OrderedDict()
        # print(para)
        for key in para.keys():
            if 'gnn' in key:
                param[key] = para[key]
        new_param = copy.deepcopy(self.model.state_dict())
        new_param.update(param)
        self.model.load_state_dict(new_param)
        

    def train_epoch(self,num_epoch):
        if self.config['pretrain']:
            self.pre_train(self.config['pre_epoch'])
        self.model = self.model.to(self.config['device'])
        
        for epoch in range(num_epoch):
            self.model.train()
            loss_epoch = 0
            for iter, (batched_graph, labels) in enumerate(self.train_loader):
                batched_graph = batched_graph.to(self.config['device'])
                
                labels = labels.to(self.config['device'])

                feats = batched_graph.ndata.pop('feat')
                weights = batched_graph.edata.pop('weight')
                
                outputs = self.model(batched_graph, feats, weights)
                loss = self.loss_func(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss_epoch += loss.item()*labels.shape[0]
            
            self.scheduler.step()
                
            loss_epoch /= len(self.train_loader.dataset)    

            train_eval = self.evaluate(self.train_loader, self.model)
            print(f"client_{self.config['client_id']:02d}: epoch:{epoch:3d} train/loss={loss_epoch:.5f}, train/eval={train_eval:.5f}", end='')
            valid_eval = self.evaluate(self.val_loader, self.model)
            print(f", val/eval={valid_eval:.5f}")
            if self.config['hpo']:
                nni.report_intermediate_result(valid_eval)

        self.model = self.model.cpu()
        
    def evaluate_all(self):
        self.model = self.model.to(self.config['device'])
        valid_eval = self.evaluate(self.val_loader, self.model)
        print(f"client_{self.config['client_id']:02d}: val/eval={valid_eval:.5f}")
        self.model = self.model.cpu()
        if self.config['hpo']:
            nni.report_final_result(valid_eval)
        return valid_eval

    def evaluate_cls(self, dataloader, model):
        model.eval()
        test_pred = []
        test_label = []
        
        with torch.no_grad():    
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(self.config['device'])
                labels = labels.view(-1)
                
                feats = batched_graph.ndata.pop('feat')
                weights = batched_graph.edata.pop('weight')
                
                outputs = model(batched_graph, feats, weights)
                pred = torch.softmax(outputs, 1)
                pred = torch.max(pred, 1)[1].view(-1)
                test_pred += pred.detach().cpu().numpy().tolist()
                test_label += labels.cpu().numpy().tolist()
            
        err_rate = (1 - accuracy_score(test_label, test_pred))
        return err_rate

    def evaluate_reg(self, dataloader, model):
        model.eval()
        total_num = 0
        total_score = 0
        
        with torch.no_grad():    
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(self.config['device'])
                # labels = labels.to(self.config['device'])
                
                feats = batched_graph.ndata.pop('feat')
                weights = batched_graph.edata.pop('weight')
                
                outputs = model(batched_graph, feats, weights).cpu()
                score = F.mse_loss(outputs, labels)
                
                total_num += len(labels)
                total_score += score * len(labels)
            
        mse = 1.0 * total_score / total_num
        return mse.item()

    def evaluate(self,dataloader,model):
        if self.task == 'cls':
            return self.evaluate_cls(dataloader, model)
        else:
            return self.evaluate_reg(dataloader, model)
        
    def inference_all(self):
        if not self.config['train_all']:
            return 
        test_dataset = self.val_loader.dataset
        dataloader = self.val_loader
        predict_score = []
        
        self.model = self.model.to(self.config['device'])
        self.model.eval()
        with torch.no_grad():
            sample_id = 0
            for graph, labels in dataloader:
            # for sample_id, (graph, label) in enumerate(test_dataset):
                graph = graph.to(config['device'])
                
                feat = graph.ndata.pop('feat')
                weight = graph.edata.pop('weight')
                
                outputs = self.model(graph, feat, weight)
                # print(outputs.shape)
                # print(outputs[0])
                for i in range(outputs.shape[0]):
                    out = outputs[i]
                    if self.task == 'reg':
                        prediction = out.split(1, dim=0)
                        prediction = list(map(lambda x: x.item(), prediction))
                        predict_score.append([config['client_id'], sample_id, *prediction])
                    else:
                        prediction = torch.max(out, 0)[1].item()
                        predict_score.append([config['client_id'], sample_id, prediction])
                    
                    
                    sample_id+=1
                
        df = pd.DataFrame(predict_score)
        print(df)
        df.to_csv(config['prediction_path'], header=False, index=False)    
    
    
    def cross_validation(self):
        from dataset import AnalytiCupDataset
        from sklearn.model_selection import KFold
        dataset = AnalytiCupDataset(self.config['client_id'], split=['train', 'val'])
        kfold = KFold(n_splits=5, shuffle=True)
        
        init_param = self.model_param()
        
        result = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_dataset = dgl.data.utils.Subset(dataset, train_idx)
            val_dataset = dgl.data.utils.Subset(dataset, val_idx)
            
            self.train_loader = GraphDataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                drop_last=False,
                shuffle=True
            )
            self.val_loader = GraphDataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                drop_last=False,
                shuffle=False
            ) 
            
            self.update_param(init_param)
            self.optimizer, self.scheduler = load_optim(config, self.model)
            
            self.train_epoch(self.config['num_epoch'])
            result.append(self.evaluate_all())
            
        print(f"{np.mean(result):.4f} ± {np.std(result):.4f}")
        if self.config['hpo']:
            nni.report_final_result(np.mean(result))
            
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('client_id', type=int)
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--hpo', action='store_true')
    parser.add_argument('-cv', '--cross-valid', action='store_true')
    
    args = parser.parse_args().__dict__
    
    config = read_config('./Fed_learn/common_config.json')
    if args['cfg'] is not None:
        config.update(read_config(args['cfg']))
    config.update(args)
    if args['hpo']:
        config.update(nni.get_next_parameter())
    setup_cfg(config)
    print(config)
    seed_everything(config['seed'])
    
    if config['cross_valid']:
        client = Client(config)
        client.cross_validation()
    else:
        client = Client(config)
        client.train_epoch(config['num_epoch'])
        client.evaluate_all()
        client.inference_all()