import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import GINEConv, GraphConv, APPNPConv, EGATConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling, SumPooling, AvgPooling, MaxPooling, Set2Set
from dgl.nn.pytorch.utils import JumpingKnowledge

class MLP(nn.Module):
    # 两层MLP+BN层
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = [
            nn.Linear(in_dim, 2*in_dim),
            nn.BatchNorm1d(2*in_dim),
            nn.ReLU(),
            nn.Linear(2*in_dim, out_dim)
        ]
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
        

class Net(nn.Module):
    # 整体网络架构，输出任务预测
    def __init__(self, 
                 num_node_emb_list, 
                 num_edge_emb_list, 
                 emb_dim=128, 
                 num_layers=5, 
                 readout='mean', 
                 JK='cat',
                 APPNP=False,
                 node_degree=True,
                 dropout=0.5,
                 n_tasks=1):
        super(Net, self).__init__()
        self.gnn = Model(num_node_emb_list, num_edge_emb_list, emb_dim, num_layers, JK, node_degree, dropout) # 提取节点表示
        node_emb_dim = emb_dim * num_layers if JK == 'cat' else emb_dim # 节点特征维度
        self.global_pool = GlobalPool(node_emb_dim=node_emb_dim, readout=readout) # 全局池化
        if APPNP:
            self.appnp = APPNPConv(k=5, alpha=0.8)
        self.downstream_network = nn.Linear(node_emb_dim, n_tasks) # 下游任务网络
        
    def forward(self, g, x, w):
        node_feats = self.gnn(g, x, w)
        if hasattr(self, 'appnp'):
            node_feats = self.appnp(g, node_feats)
        graph_feats = self.global_pool(g, node_feats)
        return self.downstream_network(graph_feats)
    
    
class Model(nn.Module):
    def __init__(self, 
                 num_node_emb_list, 
                 num_edge_emb_list, 
                 emb_dim, 
                 num_layers, 
                 mode,
                 node_degree,
                 dropout):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.mode = mode
        self.node_embedding = CIKMEmbedding(emb_dim, num_node_emb_list, degree=node_degree)
        
        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                # 最后一层不需要加非线性函数
                self.gnn_layers.append(
                    ConvBlock(emb_dim=emb_dim, 
                              dropout=dropout,
                              edge_embedding=CategoricalEmbedding(emb_dim, num_edge_emb_list)
                              ))
            else:
                self.gnn_layers.append(
                    ConvBlock(emb_dim=emb_dim, 
                              activation=F.relu, 
                              dropout=dropout,
                              edge_embedding=CategoricalEmbedding(emb_dim, num_edge_emb_list)
                              ))
        if self.mode in ['cat', 'max', 'lstm']:                
            self.jk = JumpingKnowledge(mode, emb_dim, num_layers)  
        
    def forward(self, g, x, w):
        node_feats = self.node_embedding(g, x)
        all_layer_node_feats = [node_feats]
        vn_emb = None
        for layer in range(self.num_layers):
            node_feats, vn_emb = self.gnn_layers[layer](g, all_layer_node_feats[layer], w, vn_emb)
            all_layer_node_feats.append(node_feats)
        
        if self.mode in ['cat', 'max', 'lstm']:
            node_feats = self.jk(all_layer_node_feats[1:])
        elif self.mode == 'last':
            node_feats = all_layer_node_feats[-1]
        return node_feats
        
    
class CIKMEmbedding(nn.Module):
    def __init__(self, emb_dim, num_node_emb_list, degree=False):
        super().__init__()
        # Embedding节点的类别特征 
        self.node_embedding = CategoricalEmbedding(emb_dim, num_node_emb_list)
        # Embedding节点的度数
        if degree:
            self.degree_embedding = nn.Embedding(64, emb_dim)
        
    def forward(self, g, x, perturb=None):
        h = self.node_embedding(x)
        if hasattr(self, "degree_emb"):
            h = h + self.degree_embedding(g.in_degrees())
        if perturb is not None:
            h = h + perturb
        return h
    
    
class GlobalPool(nn.Module):
    # 读出图级别的表示
    def __init__(self, node_emb_dim, readout='sum'):
        super().__init__()
        
        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'attention':
            self.readout = GlobalAttentionPooling(gate_nn=MLP(node_emb_dim, 1))
        elif readout == 'set2set':
            self.readout = Set2Set(node_emb_dim, 2, 1)
        else:
            raise ValueError("Expect readout to be 'sum', 'mean', 'max', 'attention' or 'set2set', got {}".format(readout))
        
    def forward(self, g, x):
        return self.readout(g, x)
            

class CategoricalEmbedding(nn.Module):
    # 类别特征Embedding
    def __init__(self, emb_dim, num_emb_list):
        # num_emb_list为特征类别个数的列表
        super().__init__()
        self.embeddings = nn.ModuleList()
        # 沿着特征维度依次Embedding
        for num_emb in num_emb_list:
            emb_module = nn.Embedding(num_emb, emb_dim)
            self.embeddings.append(emb_module)
        
    def forward(self, categorical_feats):
        embeds = []
        # 将类别特征划分为 list of LongTensor of shape(N)
        categorical_feats = categorical_feats.long().split(1, dim=-1)
        categorical_feats = [_.squeeze() for _ in categorical_feats]
        for i, feats in enumerate(categorical_feats):
            embeds.append(self.embeddings[i](feats))
        embeds = torch.stack(embeds, dim=0).sum(0)
        return embeds
    

class ConvBlock(nn.Module):
    # 层级的gin卷积块
    def __init__(self, emb_dim, dropout=0.5, activation=None, conv_type='gin', edge_embedding=None):
        super().__init__()
        self.edge_embedding = edge_embedding
        self.conv_type = conv_type
        self.activation = activation
        
        self.virtual_node = VirtualNode(emb_dim, emb_dim, residual=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        if conv_type == 'gin':
            self.conv = GINEConv(MLP(emb_dim, emb_dim), learn_eps=True)
        else:
            raise ValueError(f"{conv_type} is not in ConvBlock conv type!")
        
    def forward(self, g, x, w, vn_emb=None):
        h = x
        # use virtual node to update node embedding
        h, vn_emb = self.virtual_node.update_node_emb(g, h, vn_emb)
        
        # Conv Block
        if self.conv_type == 'gin':
            h = self.conv(g, h, self.edge_embedding(w))
        h = self.batch_norm(h)
        if self.activation is not None:
            h = self.activation(h)
        h = self.dropout(h)
        
        # use updated node embedding to update virtual node embeddding
        vn_emb = self.virtual_node.update_vn_emb(g, h, vn_emb)
        return h, vn_emb    
    

class VirtualNode(nn.Module):
    # 修改了gtrick中dgl版本的BUG
    def __init__(self, in_feats, out_feats, dropout=0.5, residual=False):
        '''
            in_feats (int): Feature size before conv layer.
            out_feats (int): Feature size after conv layer.
            dropout (float): Dropout rate on virtual node embedding. Defaults: 0.5.
            residual (bool): If True, use residual connection. Defaults: False.
        '''

        super().__init__()
        self.dropout = dropout
        # Add residual connection or not
        self.residual = residual

        # Set the initial virtual node embedding to 0.
        self.vn_emb = nn.Embedding(1, in_feats)
        # nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        if in_feats == out_feats:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(in_feats, out_feats)

        # MLP to transform virtual node at every layer
        self.mlp_vn = nn.Sequential(
            nn.Linear(out_feats, 2 * out_feats),
            nn.BatchNorm1d(2 * out_feats),
            nn.ReLU(),
            nn.Linear(2 * out_feats, out_feats),
            nn.BatchNorm1d(out_feats),
            nn.ReLU())

        self.pool = dgl.nn.SumPooling()

        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.linear, nn.Identity):
            self.linear.reset_parameters()

        for c in self.mlp_vn.children():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
        nn.init.constant_(self.vn_emb.weight.data, 0)

    def update_node_emb(self, g, x, vx=None):
        # Virtual node embeddings for graphs
        if vx is None:
            vx = self.vn_emb(
                torch.zeros(g.batch_size).long().to(x.device))

        if g.batch_size > 1:
            batch_id = dgl.broadcast_nodes(g, torch.arange(
                g.batch_size).to(x.device).view(g.batch_size, -1)).flatten()
        else:
            batch_id = 0

        # Add message from virtual nodes to graph nodes
        h = x + vx[batch_id]
        return h, vx

    def update_vn_emb(self, g, x, vx):
        # Add message from graph nodes to virtual nodes
        vx = self.linear(vx)
        vx_temp = self.pool(g, x) + vx

        # transform virtual nodes using MLP
        vx_temp = self.mlp_vn(vx_temp)

        if self.residual:
            vx = vx + F.dropout(
                vx_temp, self.dropout, training=self.training)
        else:
            vx = F.dropout(
                vx_temp, self.dropout, training=self.training)

        return vx

if __name__ == "__main__":
    print('='*20,"GNN",'='*20)