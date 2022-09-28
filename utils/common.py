import json
import os
import random
import numpy as np
import torch
import dgl

def read_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as fr:
        config = json.load(fr)
    return config

def setup_cfg(cfg):
    # set device
    if torch.cuda.is_available():
        cfg['device'] = f"cuda:{cfg['gpu']}"
    else:
        cfg['device'] = "cpu"
    # set path    
    if 'prediction_path' not in cfg.keys():
        cfg['prediction_path'] = os.path.join(cfg['save_path'], f"prediction/submit_{cfg['client_id']}.csv")
    cfg['model_parameter_path'] = os.path.join(cfg['save_path'], f"model/{cfg['client_id']}.pkl")
    cfg['result_path'] = os.path.join(cfg['save_path'], f"result/{cfg['client_id']}.txt")
    
def seed_everything(seed=42):
    # basic
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # dgl
    dgl.seed(seed)
    dgl.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)