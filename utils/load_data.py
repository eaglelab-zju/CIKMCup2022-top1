from dataset import AnalytiCupDataset
from dgl.dataloading import GraphDataLoader

def load_data(cfg):
    if cfg['train_all']:
        # 将训练集和验证集全部用作训练
        train_dataset = AnalytiCupDataset(
            client_id=cfg['client_id'], 
            split=['train', 'val'], 
            self_loop=cfg['self_loop'],
            add_attr=cfg['add_attr']
            )
        # 测试集标签默认为0
        val_dataset = AnalytiCupDataset(
            client_id=cfg['client_id'], 
            split=['test'], 
            self_loop=cfg['self_loop'],
            add_attr=cfg['add_attr']
            )
    else:
        train_dataset = AnalytiCupDataset(
            client_id=cfg['client_id'], 
            split=['train'], 
            self_loop=cfg['self_loop'],
            add_attr=cfg['add_attr']
            )
        val_dataset = AnalytiCupDataset(
            client_id=cfg['client_id'], 
            split=['val'], 
            self_loop=cfg['self_loop'],
            add_attr=cfg['add_attr']
            )
    if cfg['batch_size'] == 0:
        cfg['batch_size'] = train_dataset.num_graphs
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        drop_last=False,
        shuffle=True
    )
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        drop_last=False,
        shuffle=False
    ) 
    return train_loader, val_loader