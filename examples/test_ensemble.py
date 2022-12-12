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

device = torch.device('cuda')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/ensemble.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--p", type=str, help="the saved file path")
    return parser.parse_args()

def verify(cfg, nworker=2):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
    models = []
    weights = []
    print("Model's Order: ")
    for model_path in cfg['weight_path']:
        print("\t",model_path, end="-->")
        if cfg['module'][model_path].find('se') < 0:
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: 
            backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        print("\t",cfg['weight_path'][model_path])
        backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        backbone.eval()
        models.append(backbone)
        weights += [cfg["ensemble_weights"][model_path]]
    print("Ensemble weights: ", *weights)
    
    with torch.no_grad():
        for x in cfg['valid_data']:
            print("\nValidate...", x)
            accs, best_thresholds, eers = ensem_evaluate_model(models, valid_set[x], device=device, mode=cfg['mode'], weights=weights)
            # print('\t--{}\'s max accuracy list: {:.5f} \t best threshold: {} \teer: {:.5f}'.format(x,max(accs), thrs, eer))
            print("--accs: ", accs)
            print("--best_thresholds: ", best_thresholds)
            print("--eers: ", eers)


if __name__ == "__main__":
    args = get_args()

    with open(args.c, 'r') as file:
        print(args.c)
        config = yaml.load(file, Loader=yaml.Loader)
    verify(config)