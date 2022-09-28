import torch

def load_optim(cfg, model:torch.nn.Module):
    "return optimizer & scheduler"
    if cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError(f"optimizer type {cfg['optimizer']} not exist!")        
    
    if cfg['scheduler'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif cfg['scheduler'] == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['num_epoch'])
    else:
        raise ValueError(f"scheduler type {cfg['scheduler']} not exist!")        
    
    return optimizer, scheduler