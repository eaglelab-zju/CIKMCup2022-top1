import os
from tqdm import tqdm

import torch

import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.data import DGLDataset
from collections import namedtuple, Counter
import torch.nn.functional as F 



class AnalytiCupDataset(DGLDataset):
    """CIKM 2022 AnalytiCup Competition: Federated Hetero-Task Learning
    每个client的Dataset类都相互独立，即AnalytiCupDataset每次只能加载一个client，避免了数据泄露。
    """
    def __init__(self, 
                 client_id:int,
                 split=['train', 'val'],
                 self_loop=True,
                 add_attr = True,
                 raw_dir='./save/data/raw', 
                 save_dir='./save/data/.dgl',
                 force_reload=False,
                 verbose=False):
        self.client_id = client_id
        self.split = split
        self.self_loop = self_loop
        self.add_attr = add_attr

        super(AnalytiCupDataset, self).__init__(
            name="CIKM22AnalytiCupCompetition", 
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose
            )  
    
    def process(self):
        print(f'process {self.client_id} {str(self.split)} data!')
        # load full graph to get info
        self.dataset_full = {}
        for part in ['train', 'val', 'test']:
            print(f'Start to process {part} dataset from PyG to DGL...')
            path = os.path.join(self.raw_dir, f'{self.client_id}/{part}.pt')
            self.dataset_full[part] = self._load_data(path)
        
        # only split graph
        self.graphs, self.labels = [], []
        for part in self.split:
            part_graphs, part_labels = self.dataset_full[part]
            self.graphs.extend(part_graphs)
            self.labels.extend(part_labels)
        # 类别特征统计信息（类别个数）
        self.num_node_emb_list = self._get_num_node_emb_list(self.dataset_full)
        self.num_edge_emb_list = self._get_num_edge_emb_list(self.dataset_full)
        self.labels = torch.cat(self.labels, dim=0)
        # 默认添加自环
        if self.self_loop:
            self.add_self_loop()

        if self.add_attr:
            print('123')
            feature_dim = 128
            degrees = []
            for g in self.graphs:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 128

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")

            for g in self.graphs:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat

    
    def _load_data(self, path):
        # 导入pyg格式原始数据
        data_client = torch.load(path)
        graphs, labels = [], []
        
        for pyg_data in tqdm(data_client):
            # 读入pyg格式
            x, edge_index, edge_attr, y = pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr, pyg_data.y
            edge_index = tuple(map(lambda x: x.squeeze(), edge_index.split(1, dim=0)))
            # 转化为dgl格式
            dgl_data = dgl.graph(edge_index, num_nodes=pyg_data.num_nodes)
            dgl_data.ndata['feat'] = x
            if edge_attr is not None:
                dgl_data.edata['weight'] = edge_attr
            else:
                dgl_data.edata['weight'] = torch.ones((dgl_data.num_edges(),1)) # 无边特征则为全1
            
            graphs.append(dgl_data)
            labels.append(y.view(1, -1))

        return graphs, labels
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.graphs)
    
    @property
    def num_tasks(self):
        if self.client_id < 9:
            return 2 # cls
        else:
            return self.labels.shape[-1] # reg
    
    @property
    def num_graphs(self):
        return len(self.graphs)
    
    @property
    def nfeat_dim(self):
        return self.graphs[0].ndata['feat'].shape[-1]
    
    @property
    def efeat_dim(self):
        if 'weight' in self.graphs[0].edata:
            return self.graphs[0].edata['weight'].shape[-1]
        else:
            return None
    
    def _get_num_node_emb_list(self, dataset_full:dict):
        # 节点特征统计信息（类别个数）
        item_list = torch.zeros(self.nfeat_dim)
        for part in ['train', 'val', 'test']:
            for graph in dataset_full[part][0]:
                x = graph.ndata['feat']
                mx, _ = torch.max(x, dim=0)
                item_list = torch.max(item_list, mx)
            
        item_list += 1
        item_list = item_list.split(1, dim=0)
        item_list = [int(item_num.item()) for item_num in item_list]
        return item_list
    
    def _get_num_edge_emb_list(self, dataset_full:dict):
        # 边特征统计信息（类别个数）
        if self.efeat_dim is None:
            return None
        
        item_list = torch.zeros(self.efeat_dim)
        for part in ['train', 'val', 'test']:
            for graph in dataset_full[part][0]:
                w = graph.edata['weight']
                mx, _ = torch.max(w, dim=0)
                item_list = torch.max(item_list, mx)
            
        item_list += 1
        item_list = item_list.split(1, dim=0)
        item_list = [int(item_num.item()) for item_num in item_list]
        return item_list
    
    def save(self):
        # 保存dgl格式数据到本地
        for part in ['train', 'val', 'test']:
            graphs, labels = self.dataset_full[part]
            labels = torch.cat(labels, dim=0)
            # save graphs & labels
            graph_path = os.path.join(self.save_dir, f'client#{self.client_id}_{part}_dgl_graph.bin')
            save_graphs(graph_path, graphs, {'labels': labels})
        # save other info
        info_path = os.path.join(self.save_dir, f'client#{self.client_id}_info.pkl')    
        save_info(info_path, {'num_node_emb_list': self.num_node_emb_list, 'num_edge_emb_list': self.num_edge_emb_list})
        
    def load(self):
        # 从本地dgl格式数据读入
        self.graphs, self.labels = [], []
        for part in self.split:
            # load graphs & labels
            graph_path = os.path.join(self.save_dir, f'client#{self.client_id}_{part}_dgl_graph.bin')
            part_graphs, label_dict = load_graphs(graph_path)
            part_labels = label_dict['labels']
            self.graphs.extend(part_graphs)
            self.labels.extend(part_labels)
        self.labels = torch.cat(self.labels, dim=0).view(len(self.graphs), -1)
        # load other info   
        info_path = os.path.join(self.save_dir, f'client#{self.client_id}_info.pkl')
        self.num_node_emb_list = load_info(info_path)['num_node_emb_list']    
        self.num_edge_emb_list = load_info(info_path)['num_edge_emb_list']  
        
        if self.self_loop:
            self.add_self_loop()
        if self.add_attr:
            # print('123')
            feature_dim = 128
            degrees = []
            for g in self.graphs:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 128

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")

            for g in self.graphs:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        
    def has_cache(self):
        # 判断本地是否有dgl格式的缓存数据
        has_graphs = True
        for part in self.split:
            graph_path = os.path.join(self.save_dir,  f'client#{self.client_id}_{part}_dgl_graph.bin')
            has_graphs &= os.path.exists(graph_path)
        info_path = os.path.join(self.save_dir, f'client#{self.client_id}_info.pkl')
        return has_graphs and os.path.exists(info_path)
    
    def add_self_loop(self):
        for i, graph in enumerate(self.graphs):
            self.graphs[i] = graph.remove_self_loop().add_self_loop()
    
    
# test module    
import unittest
class TestDataset(unittest.TestCase):
    def test_is_all_feat_categorical(self):
        print('test node & edge feat is categorical...')
        
        for client_id in range(1, 14):
            print(f"clinet#{client_id}", end=', ')
            dataset = AnalytiCupDataset(client_id=client_id, split=['test'], raw_dir='/home/fangzeyu/projects/CIKM22Competition/data/raw/')
            
            for graph, label in dataset:
                x = graph.ndata['feat']
                assert(torch.equal(x - x.long(), torch.zeros_like(x)))
                assert(torch.equal(torch.min(x), torch.tensor(0)))
                
                w = graph.edata.get('weight', None)
                if w is not None:
                    assert(torch.equal(w - w.long(), torch.zeros_like(w)))
                    assert(torch.equal(torch.min(w), torch.tensor(0)))
            
            print(dataset.nfeat_dim, dataset.efeat_dim, dataset.num_node_emb_list, dataset.num_edge_emb_list if dataset.efeat_dim is not None else None)
            print("passed the test!")
    
    def test_refactor_dataset(self):
        print('<pref>(dataset.py): self loop, save & load, graphs & info')
        
        for client_id in range(1, 14):
            print(f"clinet#{client_id}", end=', ')
            train_dataset = AnalytiCupDataset(client_id=client_id, split=['train'], verbose=True)
            val_dataset = AnalytiCupDataset(client_id=client_id, split=['val'], verbose=True)
            test_dataset = AnalytiCupDataset(client_id=client_id, split=['test'], verbose=True)
            
            all_label_dataset = AnalytiCupDataset(client_id=client_id, split=['train', 'val'])
            
            graph = train_dataset[0][0]
            g = dgl.to_bidirected(graph)
            print(graph, g)
            
            print('feat dim:', train_dataset.nfeat_dim, train_dataset.efeat_dim)
            print(train_dataset.num_node_emb_list, train_dataset.num_edge_emb_list)
            print('num graphs:', train_dataset.num_graphs, val_dataset.num_graphs, test_dataset.num_graphs)

    
if __name__ == "__main__":
    unittest.main()
    