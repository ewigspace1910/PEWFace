
from __future__ import print_function, absolute_import
from asyncio import queues
from asyncore import write
from concurrent.futures import process
from math import dist
import os
import sys
sys.path.append(os.getcwd())
import argparse
import yaml
import time
import torch
import tqdm
import numpy as np
from modules.evaluate_ensemble import  only_voting
from modules.evaluate import evaluate_model, load_model
from modules.models import Backbone
from modules.dataloader import get_DataLoader, ValidDataset
from examples.parallel_cpu import single_verify, seq_ensemble_verify


import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Queue, JoinableQueue


device = torch.device("cuda") if torch.cuda.is_available()  else None
if device is None: assert False, "No GPU" 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/ensemble/soft.yaml', help='config path')
    parser.add_argument('--b', type=int, default=16, help='batchsize')
    parser.add_argument('--l', type=int, default=5, help='number loop')
    parser.add_argument('--n', type=int, default=4, help='number workers')
    parser.add_argument("--e", action='store_true', help="using ensemble or single model")
    parser.add_argument("--parallel", action='store_true', help="using parallel computation")
    return parser.parse_args()


def writer(source, queues, device=None):
    '''Args:
    - source : Dataloader
    - device : CPU/CUDa
    - queues: queues list, each queue corresponding a model
    '''
    for img1, img2, _ in tqdm.tqdm(source):
        for q in queues:
            q.put((img1.clone().to(device), img2.clone().to(device)))
    for q in queues: q.put((None, None))


def pinfer_quueue(DSqueue, model, return_dict):
    dist = np.array([])
    while True:
        img1, img2 = DSqueue.get()
        if img1 is None: 
            print("break!!!!")
            break
        embd_1 = model(img1)
        embd_2 = model(img2)
        diff = torch.sum((embd_1-embd_2) ** 2, axis=1).cpu().data.numpy()
        dist = np.hstack((dist, diff))
        del embd_1, embd_2, img1, img2
        torch.cuda.empty_cache()
    return_dict.append(dist)
    print(return_dict)

def pinfer(DSqueue, model, return_dict):
    dist = np.array([])
    while len(DSqueue) > 0:
        img1, img2  = DSqueue.pop()
        embd_1 = model(img1)
        embd_2 = model(img2)
        diff = torch.sum((embd_1-embd_2) ** 2, axis=1).cpu().data.numpy()
        dist = np.hstack((dist, diff))
        # del embd_1, embd_2, img1, img2
        # torch.cuda.empty_cache()
    return_dict.append(dist)

def parallel_ensemble_verify(cfg, nworker=2, device=None):
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x], only_path=False)
        valid_set[x] = get_DataLoader(valid_dataset,
                                batch_size=cfg['batch_size'],
                                shuffle=False,
                                num_workers=nworker)
    models = []
    for model_path in cfg['weight_path']:
        print("--->", model_path)
        if cfg['module'][model_path].find('se') < 0: backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else:  backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
        backbone = load_model(backbone, cfg['weight_path'][model_path])
        backbone = backbone.to(device)
        backbone.eval()
        models.append(backbone)
    
    costs = {}
    with torch.no_grad():
        for x in cfg['valid_data']:
            global dists
            print("\nValidate...", x)
            DS = [[]] * len(models)
            labels = np.array([], dtype=np.int32)
            for ii, (img1,img2, l) in tqdm.tqdm(enumerate(valid_set[x])): 
                labels = np.hstack((labels, l))
                for i in range(len(models)):
                    if len(DS[i]) ==0: DS[i] = [(img1.clone().to(device), img2.clone().to(device))]
                    else: DS[i].append((img1.clone().to(device), img2.clone().to(device)))
                if ii == 125: break 

            #####parallel seting#####
            
            processes, dsqueues = [], []
            endists = [np.array([])] * len(models) #distants
            manager = multiprocessing.Manager()
            return_dict = manager.list()
            
            for i, model in enumerate(models):
                # q = JoinableQueue() #Queue()
                p = multiprocessing.Process(target=pinfer, args=(DS[i], model, return_dict,))
                # p.daemon = False
                processes += [p]
                # dsqueues += [q]
            ########################
            for p in processes: p.start()
            tstart  = time.time()
            # writer(source=DS , queues=dsqueues , device=device)
            for p in processes: p.join()
            endists = np.array(return_dict)
            accs,_, _ = only_voting(endists, labels, device=device, mode=cfg['mode'])
            elapsed = time.time() - tstart
            #########################
            print("--accs: ", accs)
            costs[x] = [elapsed]
            #free ram
            del DS
            torch.cuda.empty_cache()
            time.sleep(5)

    return costs


#==============================

if __name__ == "__main__":
    args = get_args()

    time_costs = {}
    with open(args.c, 'r') as file:
        print(args.c)
        config = yaml.load(file, Loader=yaml.Loader)
        config['batch_size'] = args.b        
    costs = {}
    nloop = args.l
    if args.parallel:             
            try: multiprocessing = multiprocessing.get_context('spawn')
            except: pass
    for i in range(nloop): 
        
        if not args.e : 
            cost = single_verify(config, nworker=args.n, device=device)
        elif args.parallel: 
            cost = parallel_ensemble_verify(config, nworker=args.n, device=device)
        else: 
            cost = seq_ensemble_verify(config, nworker=args.n, device=device)
        
        #save statistic
        if i == 0: costs = cost
        else: 
            for x in cost: costs[x]+=cost[x]
        time.sleep(15)
    #print
    for x in costs:
        if not args.e : 
            print("monitoring single verification runtime")
        elif args.parallel: 
            print("monitoring parallel verification runtime")
        else: 
            print("monitoring sequencently verification runtime")
        print("Total of time for GPU infering on {}: {}s".format(x, sum(costs[x])/nloop))

