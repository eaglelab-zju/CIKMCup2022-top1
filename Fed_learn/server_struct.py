# 联邦学习只学习图结构的embedding
# 服务端和客户端共享一个模型
import sys
sys.path.append('.')

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.models import build_model
from tqdm import tqdm
import numpy as np
import torch
from client_struct import Client_struct
import torch.nn as nn
import copy
from utils import seed_everything

class Server_struct:
    def __init__(self,args):
        self.client_num = 13
        self.client_list = []
        for i in range(1,self.client_num+1):
            # 每个客户端使用相同的模型
            print(f'init client {i}')
            self.client_list.append(Client_struct(i,args))
        # 服务端模型也相同
        self.model:nn.Module = build_model(args)

    def load_model(self,model_param):
        self.model.load_state_dict(model_param)
    
    def save_param(self):
        return copy.deepcopy(self.model.state_dict())

    def train(self,epoch_num):
        epoch_iter = tqdm(range(epoch_num))
        for epoch in epoch_iter:
            cl = np.arange(self.client_num)
            #随机生成该轮epoch中的客户端训练顺序
            np.random.shuffle(cl)
            loss_list = []
            # 客户端依次训练
            for id in cl:
                #保存服务端模型参数
                para = self.save_param()
                #客户端更新参数
                self.client_list[id].load_model(para)
                #客户端训练
                loss_list.append(self.client_list[id].train_epoch())
                #保留客户端参数
                para = self.client_list[id].save_param()
                #更新服务端参数
                self.load_model(para)
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
        #保存最终训练出来的模型
        torch.save(self.model.state_dict(), 'save/s_model/s_model')
        torch.save(self.model, 'save/s_model/s_model.pkl')

# python Fed_learn/server_struct.py --device 0
if __name__ == '__main__':
    args = build_args()
    args.num_features = 128
    args.norm = 'batchnorm'
    args.loss_fn = "sce"
    args.encoder = 'gin'
    args.decoder = 'gin'
    print(args)
    seed_everything(42)
    server = Server_struct(args)
    server.train(args.max_epoch)
