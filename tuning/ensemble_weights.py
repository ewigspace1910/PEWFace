import os
import argparse
import yaml
import tqdm
import torch
import numpy as np
import sys
sys.path.append(os.getcwd())
from modules.evaluate_ensemble import ensem_evaluate_model, load_model
from modules.models import Backbone
from modules.dataloader import get_DataLoader, ValidDataset
import optuna

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
NWORKER = 2
CONFIG = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/ensemble.yaml', help='config path')
    parser.add_argument("--n", type=int, default=4, help="the number of workers")
    parser.add_argument("--t", type=int, default=100,help="A number of Trial")
    parser.add_argument("--p", action='store_true',help="Using pruner or not")
    parser.add_argument("--m", type=int, default=1,help="objective metrics: 1-accuracy, 2-eer")    
    parser.add_argument("--cpu",action="store_true",help="using CPU instead GPU")
    parser.add_argument("--d", type=str, help="optimized directions including: maximize, minimize", choices=["maximize", "minimize"])
    return parser.parse_args()

def verify(cfg):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=NWORKER)
    models = []
    weights = []
    for model_path in cfg['weight_path']:
        if cfg['module'][model_path].find('se') < 0: backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        backbone.eval()
        models.append(backbone)
        weights += [cfg["ensemble_weights"][model_path]]
    #print("\t--->Ensemble weights: ", *weights)
    mean_acc = []
    with torch.no_grad():
        for x in cfg['valid_data']:
            accs, _, eers = ensem_evaluate_model(models, valid_set[x], 
                            device=device, mode=cfg['mode'], weights=weights,
                            flag_monitor=False)
            mean_acc += [accs[-1]] if cfg["ObjMetric"] == 1 else [eers[-1]]
    return sum(mean_acc) / len(mean_acc)

def objective(trial):
    assert not CONFIG is None, "CONFIG is None!"
    for k in CONFIG['weight_path']:
        CONFIG['ensemble_weights'][k] = trial.suggest_float('W-{}'.format(k), 0.05, 1)
    mean = verify(cfg=CONFIG)
    return mean



if __name__ == "__main__":
    args = get_args()
    print("----NEWRUN----- \n\t---> CFGFile :{}".format(args.c))
    with open(args.c, 'r') as file:
        CONFIG = yaml.load(file, Loader=yaml.Loader)
    NWORKER = args.n
    CONFIG["ObjMetric"] = args.m
    device = torch.device("cpu") if args.cpu else device 

    pruner = (optuna.pruners.MedianPruner() if args.p else optuna.pruners.NopPruner())
    direction = "maxi"
    study = optuna.create_study(direction = 'maximize', pruner=pruner)
    study.optimize(objective, n_trials=args.t)
    
    trial = study.best_trial
    print('Objective metric: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
