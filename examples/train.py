import sys
import os
sys.path.insert(0,os.getcwd())
import argparse
import torch
import yaml
import tqdm
import time
from torch.nn import DataParallel
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from modules.models import Backbone, OriginalBackbone
from modules.dataloader import get_DataLoader, TrainDataset, ValidDataset
from modules.metrics import CosFace, ArcFace, MagFace
from modules.metrics import ElasticArcFace, ElasticCosFace, AdaFace, AdaCos
from modules.evaluate import evaluate_model, load_model
#from modules.utils import set_memory_growth

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#set_memory_growth()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='./configs/res50.yaml', help='config path')
    parser.add_argument("--n", type=int, default=2, help="the number of workers")
    parser.add_argument("--p", type=str, default="",help="the saved model path")
    return parser.parse_args()

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.module.state_dict(), save_name)
    return save_name

def main(cfg, n_workers=2, model_file=""):
    #setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #setup path
    save_path = os.path.join(os.getcwd(), "save", cfg['model_name'])
    ckpt_path = os.path.join(save_path, "ckpt")
    log_path  = os.path.join(save_path, "log")
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)
    if not os.path.exists(log_path): os.makedirs(log_path)
    #train data
    train_dataset = TrainDataset(data_list_file=cfg['train_data'], is_training=True, input_shape=(3, cfg['image_size'], cfg['image_size']))
    trainloader = get_DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=n_workers)
    #valid data
    valid_set = {}
    for x in cfg['valid_data']:
        valid_dataset = ValidDataset(data_list_file=cfg['valid_data'][x])
        valid_set[x] = get_DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=n_workers)

    #get backbone
    if cfg['backbone'].lower() == 'resnet50':
        if cfg['module'] == 'ir': backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: backbone = Backbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
    elif cfg['backbone'].lower() == 'orginalr50':
        if cfg['module'] == 'ir': backbone = OriginalBackbone(50, drop_ratio=cfg['drop_ratio'], mode='ir')
        else: backbone = OriginalBackbone(50, drop_ratio=cfg['drop_ratio'], mode='ir_se')
    else:
        assert False, "Backbone must be either resnet50 or originalr50"

    #metrics
    margin = True
    print("using {}".format(cfg['loss'].lower()))
    if cfg['loss'].lower() == 'cosloss':
        header = CosFace(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'arcloss':
        header = ArcFace(in_features=cfg['embd_size'],
                                out_features=cfg['class_num'],
                                s=cfg['logits_scale'], m=cfg['logits_margin'])
    elif cfg['loss'].lower() == 'magloss':
        assert len(set(['l_a', 'u_a', 'l_m', 'u_m', 'lambda_g']) & set(cfg.keys())) == 5, "Missing some MagFace hyperparameters"
        header = MagFace(in_features=cfg['embd_size'], 
                                out_features=cfg['class_num'], 
                                s=cfg['logits_scale'], 
                                l_a=cfg['l_a'], u_a=cfg['u_a'], l_m=cfg['l_m'], u_m=cfg['u_m'], lambda_g=cfg['lambda_g'])
    elif cfg['loss'].lower() == 'elastic-cos':
        assert len(set(['logits_scale', 'logits_margin', 'logits_std', 'logits_plus']) & set(cfg.keys())) == 4, "Missing some ElasticFace hyperparameters"
        header = ElasticCosFace(in_features=cfg['embd_size'],
                            out_features=cfg['class_num'],
                            s=cfg['logits_scale'],
                            m=cfg['logits_margin'],
                            std=cfg['logits_std'],
                            plus=cfg['logits_plus'])
    elif cfg['loss'].lower() == 'elastic-arc':
        assert len(set(['logits_scale', 'logits_margin', 'logits_std', 'logits_plus']) & set(cfg.keys())) == 4, "Missing some ElasticFace hyperparameters"
        header = ElasticArcFace(in_features=cfg['embd_size'],
                    out_features=cfg['class_num'],
                    s=cfg['logits_scale'],
                    m=cfg['logits_margin'],
                    std=cfg['logits_std'],
                    plus=cfg['logits_plus'])
    elif cfg['loss'].lower() == 'adacos':
        # assert len(set(['logits_scale', 'logits_margin', 'logits_h', 't_alpha']) & set(cfg.keys())) == 4, "Missing some ElasticFace hyperparameters"
        header = AdaCos(in_features=cfg['embd_size'],
                    out_features=cfg['class_num'],
                    m=cfg['logits_margin'])

    elif cfg['loss'].lower() == 'adaface':
        assert len(set(['logits_scale', 'logits_margin', 'logits_h', 't_alpha']) & set(cfg.keys())) == 4, "Missing some ElasticFace hyperparameters"
        header = AdaFace(in_features=cfg['embd_size'],
                    out_features=cfg['class_num'],
                    s=cfg['logits_scale'],
                    m=cfg['logits_margin'],
                    h=cfg['logits_h'],
                    t_alpha=cfg['t_alpha'])
    else:
        header = torch.nn.Linear(cfg['embd_size'], cfg['class_num'], bias=False)
        margin = False
    
    #data parapell
    if model_file != "": load_model(backbone, model_file)
    backbone = DataParallel(backbone.to(device))
    header = DataParallel(header.to(device))

    #optimizer
    if 'optimizer' in cfg.keys() and cfg['optimizer'].lower() == 'adam':
        optimizer = Adam([{'params': backbone.parameters()}, {'params': header.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = SGD([{'params': backbone.parameters()}, {'params': header.parameters()}],
                                    lr=cfg['base_lr'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    #LossFunction+scheduerLR
    criterion = torch.nn.CrossEntropyLoss()
    
    lr_steps = [ s for s in cfg['lr_steps']] #epochs
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=cfg['decay_rate'])

    #loop
    steps_per_epoch = cfg['sample_num'] // cfg['batch_size'] + 1
    stps = steps_per_epoch if str(cfg["step_per_save"]) == "epoch" else int(cfg['step_per_save']) 
    writer = SummaryWriter(log_path)
    for e in range(1,cfg['epoch_num']+1):
        backbone.train()
        total_loss = 0.0
        num_batchs = 0
        
        for data in tqdm.tqdm(iter(trainloader)):
            inputs, label = data
            if inputs.shape[0] < cfg['batch_size']: continue
            inputs = inputs.to(device)
            label = label.to(device).long()

            logits = backbone(inputs)
            if margin: logits = header(logits, label)
            else: logits = header(logits)

            if len(logits) == 2: #MagFace
                loss = criterion(logits[0], label) + logits[1]
                logits = logits[0]
            else: loss = criterion(logits, label)
            
            #update metrics
            total_loss += loss.item()
            num_batchs += 1
            
            #update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
            optimizer.step()
            if num_batchs % stps == 0:   
                writer.add_scalar('training loss',
                            total_loss / num_batchs,
                            (e-1) * len(trainloader) + num_batchs)
                writer.add_scalar('learning rate',
                            scheduler.get_last_lr()[0],
                            (e-1) * len(trainloader) + num_batchs)

          
        #save model ---- offline test :>    
        if e > 20: save_model(backbone,  ckpt_path, cfg['model_name'], e)
        #test
        backbone.eval()
        val_log = ""
        with torch.no_grad():
            for x in cfg['valid_data']:
                acc, _, eer = evaluate_model(backbone, valid_set[x], device=device)
                acc = max(acc)
                writer.add_scalar('verification accuracy _ {} dataset'.format(x), acc, e * num_batchs)
                writer.add_scalar('verification EER _ {} dataset'.format(x), eer, e * num_batchs)
                val_log += '\t--{}\'s acc: {:.3f} \t eer ~ {:.3f} '.format(x, acc*100, eer*100)
        
        print("Epoch: {}/{} \t--LR: {:.6f} \t--Train Loss: {:.6f} \
            || {} \
            ".format(e,cfg['epoch_num'], scheduler.get_last_lr()[0], total_loss / num_batchs, val_log))

        scheduler.step() 
                          
    writer.close()


if __name__ == '__main__':
    args = get_args()
    with open(args.c, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)
    print(config)
    main(config ,n_workers=args.n, model_file=args.p)
