import os
import argparse
import torch
import yaml
import tqdm
import time
from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from modules.models import Backbone
from modules.dataloader import get_DataLoader, TrainDataset, ValidDataset
from modules.metrics import CosMarginProduct, ArcMarginProduct, MagMarginProduct
from modules.evaluate import evaluate_model
from modules.focal_loss import FocalLoss

import optuna
#from modules.utils import set_memory_growth

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#set_memory_growth()

NWORKER = 2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    return parser.parse_args()

def main(cfg, n_workers=2):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #train data
    train_dataset = TrainDataset(data_list_file=cfg['train_data'],
                       is_training=True,
                       input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset,
                                   batch_size=cfg['batch_size'],
                                   shuffle=True,
                                  num_workers=n_workers)
    #valid data
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=n_workers)
        break

    #get backbone
    if cfg['backbone'].lower() == 'resnet50':
        print("use ir-se_Resnet50")
        backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], embedding_size=cfg['embd_size'], mode='ir_se')
    elif cfg['backbone'].lower() == 'resnet100':
        print("use ir-resnet100")
        backbone = Backbone(100, drop_ratio=cfg['drop_ratio'],embedding_size=cfg['embd_size'], mode='ir_se')
    else:
        print("backbone must resnet50, resnet100")
        exit()

    #metrics
    margin = True
    if cfg['loss'].lower() == 'cosloss':
        print("use Cos-Loss")
        partial_fc = CosMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'arcloss':
        print("use ArcLoss")
        partial_fc = ArcMarginProduct(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'magloss':
        print('use Mag-Loss')
        partial_fc = MagMarginProduct(in_features=cfg['embd_size'], 
                                out_features=cfg['class_num'], 
                                s=cfg['logits_scale'], 
                                l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20)
    else:
        print("No Additative Margin")
        partial_fc = torch.nn.Linear(cfg['embd_size'], cfg['class_num'], bias=False)
        margin = False
    
    #data parapell
    backbone = DataParallel(backbone.to(device))
    partial_fc = DataParallel(partial_fc.to(device))

    #optimizer
    if 'optimizer' in cfg.keys() and cfg['optimizer'].lower() == 'adam':
        optimizer = Adam([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = SGD([{'params': backbone.parameters()}, {'params': partial_fc.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    #LossFunction+scheduerLR
    if cfg['criterion'] == 'focal':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    lr_steps = [ s for s in cfg['lr_steps']] #epochs
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    for e in range(1,cfg['epoch_num']+1):
        backbone.train()
        for data in tqdm.tqdm(iter(trainloader)):
            inputs, label = data
            inputs = inputs.to(device)
            label = label.to(device).long()

            logits = backbone(inputs)
            if margin: logits = partial_fc(logits, label)
            else: logits = partial_fc(logits)

            if len(logits) == 2: 
                loss = criterion(logits[0], label) + logits[1]
                logits = logits[0]
            else: loss = criterion(logits, label) 
            #update weights
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
        scheduler.step() 
    #test
    backbone.eval()
    with torch.no_grad():
        for x in cfg['valid_data']:
            accs, _, _ = evaluate_model(backbone, valid_set[x], device=device)
            break
    return max(accs)

def objective(trial):
    #fixed params
    config['epoch_num'] = 3
    
    #estimator
    config['logits_scale'] = trial.suggest_categorical("logit_scale", [30, 64])
    config['optimizer']    = trial.suggest_categorical("opt", ['SGD', 'Adam'])
    config['base_lr']      = trial.suggest_loguniform("lr", 1e-3, 1e-1)
    config['drop_ratio']   = trial.suggest_float('dropout', 0.4, 0.6)
    #config['batch_size']   = 128
    config['lr_steps']     = [9, 14]
    config['criterion']    = trial.suggest_categorical("criterion", ['crossentropy', 'focal'])
    #run
    acc = main(config ,n_workers=NWORKER)
    return acc


if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    NWORKER = args.n
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials=8)
    
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
