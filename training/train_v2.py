# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR
# from DeepfakeBench.training.loss.focalLoss import FocalLoss

# from trainer.trainer_v2 import Trainer
from trainer.trainer_v2 import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from sklearn.model_selection import train_test_split


# ---------------------------------------- #
# CustomSubset class to create a subset of a dataset with the same attributes as the original dataset
from torch.utils.data import Dataset, Subset

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset
        self.indices = indices
        self.data_dict = dataset.data_dict  # Retain the data_dict attribute

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]  # Return the item at the given index in the subset
    
    def __len__(self):
        return len(self.indices)  # Return the length of the subset

    def __getattr__(self, name):
        return getattr(self.dataset, name)  # Delegate attribute access to the original dataset
# ---------------------------------------- #

parser = argparse.ArgumentParser(description='Process some paths.')
# parser.add_argument('--detector_path', type=str,get
                    # default='/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml',
                    # help='path to detector YAML file')
parser.add_argument('--detector', type=str, default = 'xception', help='detector name')
parser.add_argument("--train_dataset", nargs="+") 
# parser.add_argument("--test_dataset", nargs="+") # used as validation dataset
parser.add_argument("--tl", action="store_true", default=False) # use transfer learning for training the model
# nargs = '+' means one or more arguments, if not provided, default is None
# flags for saving checkpoint and features (output of the model) -> --no-save-feat -> save_feat = False
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True) 
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True) 
# data parallel config stuff
parser.add_argument("--ddp", action='store_true', default=False) # whether to use distributed data parallel
parser.add_argument('--local_rank', type=int, default=0) # local rank for distributed data parallel
# task target -> for logging (?)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task') 
# resume training from ckpt
parser.add_argument('--resume', type=str, help = "define the checkpoint path from which to resume the training")
# tags 
parser.add_argument('--tags', type=str, default="", help='specify the tags of current training task')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


# ---------------------------------------- #
# python train_v2.py --detector "xception" --tags "Xception_dfb_occ_TL" --tl ["occlusion" is the default dataset in the config file]
# python train_v2.py --detector "xception" --tags "Xception_dfb_no_occ_TL" --tl --train_dataset "no_occlusion" 

# python train_v2.py --detector "ucf" --tags "UCF_dfb_occ_TL" --tl
# python train_v2.py --detector "ucf" --train_dataset "occlusion" --tl --tags "UCF_dfb_occ_test_tl"
# ---------------------------------------- #

# initialize random seed for reproducibility -> fixed in the config file (1024)
def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config):
    # Only use the blending dataset class in training
    if 'dataset_type' in config and config['dataset_type'] == 'blend':
        if config['model_name'] == 'facexray':
            train_set = FFBlendDataset(config)
        elif config['model_name'] == 'fwa':
            train_set = FWABlendDataset(config)
        elif config['model_name'] == 'sbi':
            train_set = SBIDataset(config, mode='train')
        elif config['model_name'] == 'lsda':
            train_set = LSDADataset(config, mode='train')
        else:
            raise NotImplementedError(
                'Only facexray, fwa, sbi, and lsda are currently supported for blending dataset'
            )
    elif 'dataset_type' in config and config['dataset_type'] == 'pair':
        train_set = pairDataset(config, mode='train')  # Only use the pair dataset class in training
    elif 'dataset_type' in config and config['dataset_type'] == 'iid':
        train_set = IIDDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'I2G':
        train_set = I2GDataset(config, mode='train')
    elif 'dataset_type' in config and config['dataset_type'] == 'lrl':
        train_set = LRLDataset(config, mode='train')
    else: # if not specified (as in xceptionet), use the abstract dataset class
        train_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='train',
                )
    if config['model_name'] == 'lsda':
        from dataset.lsda_dataset import CustomSampler
        custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                sampler=custom_sampler, 
                collate_fn=train_set.collate_fn,
            )
    elif config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                )
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        if not config.get('dataset_type', None) == 'lrl':
            test_set = DeepfakeAbstractBaseDataset(
                    config=config,
                    mode='test',
            )
        else:
            test_set = LRLDataset(
                config=config,
                mode='test',
            )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last = (test_name=='DeepFakeDetection'), # drop the last batch if it is not full (only for DeepFakeDetection dataset)
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        # config['test_dataset'] = ['occlusion'] -> one_test_name = ['occlusion']
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders



def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam': # default optim
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )

    # ---------------------------------------- #
    # -------------- Focal Loss -------------- #
    # ---------------------------------------- #
    # check if definition is correct!!!
    elif opt_name == 'focal_loss':
        optimizer = optim.FocalLoss(
            # alpha=config['optimizer'][opt_name]['alpha'], # default: 0.25
            # gamma=config['optimizer'][opt_name]['gamma'], # default: 2
            # reduction=config['optimizer'][opt_name]['reduction'], # default: mean
            # using default valies
            alpha = 0.25,
            gamma = 2,
            reduction = 'mean'
        )
        
    # criterion = FocalLoss(alpha=0.25, gamma=2) 
    
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer','auc', 'acc', 'ap']:  
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring

def check_model_gradient(model):
    # Check if the gradient is active for the FC layer
    for name, param in model.named_parameters():
        if 'fc' in name:
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")
        else:
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")

def main():


    # ---------------------------------------- #
    # TODO:
    # error in loading the dataset : 
    # Error loading image at index 0: ././datasets/rgb\/media/data/rz_dataset/dfb_faces/occlusion/training/user_684510/simswap/obj_occlusion_3/180.png does not exist
    # check where the img is wrongly loaded in the abstract_dataset.py file

    # do i need to use the test_dataset argument? i just want to train the model... not sure what the test_dataset is used for here
    # check how tensorboard is used in the code and if its possible to remove it

    # try to train the model using the defaul values in the config file

    # check the xception.yaml file on the vm and see how the train/test dataset was defined 
    # (i might have change it to [occlusion / no_occlusion]), before were only the paper's datasets

    # test model with default values in the config file first

    # ---------------------------------------- #

    if args.detector == 'xception':
        detector_yaml = './config/detector/xception.yaml'
        weights_path = './pretrained/xception_best.pth'
        model_name = 'xception'

    elif args.detector == 'ucf':
        detector_yaml = './config/detector/ucf.yaml'
        weights_path = './pretrained/ucf_best.pth'
        model_name = 'ucf'
    else:
        raise NotImplementedError('detector {} is not implemented'.format(args.detector))


    # parse options and load config
    with open(detector_yaml, 'r') as f:
        config = yaml.safe_load(f)

    with open('./config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)

    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']

    config.update(config2)

    # local rank for distributed data parallel training (ddp) -> set to 0 if not using ddp
    # where is defined?
    config['local_rank']=args.local_rank

    if config['dry_run']: # if dry_run is set to True, the model will not be trained
        config['nEpochs'] = 0
        config['save_feat']=False
    
    print("loss_func: ",config['loss_func'])

    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    else:
        args.train_dataset = config['train_dataset']

    # if args.test_dataset:
    #     config['test_dataset'] = args.test_dataset
    # else:
    #     args.test_dataset = config['test_dataset']  

    print("train_dataset: ", args.train_dataset[0])
    gotcha = False
    if args.train_dataset[0] == 'gotcha_occlusion' or args.train_dataset[0] == 'gotcha_no_occlusion':
        print("using gotcha dataset!")
        gotcha = True
    # print("test_dataset", args.test_dataset)
    # breakpoint()

    print("frame_num: ", config['frame_num'])

    print("metric scoring: ", config['metric_scoring'])
    # print("using accuracy (acc) as metric scoring")
    # config['metric_scoring'] = 'acc'
    # print("metric scoring: ", config['metric_scoring'])

    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'

    # added control on 'task_target' in the config file
    if 'task_target' not in config:
        config['task_target'] = None
    
    # create logger
    # timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # task_str = f"_{config['task_target']}" if config['task_target'] is not None else "" 
    # logger_path =  os.path.join(
    #             config['log_dir'],
    #             # config['model_name'] + task_str + '_' + timenow
    #             args.tags + task_str + '_' + timenow
    #         )
    # os.makedirs(logger_path, exist_ok=True)
    # logger = create_logger(os.path.join(logger_path, 'training.log'))
    # logger.info('Save log to {}'.format(logger_path))

    print("log_dir: ", config['log_dir'])
    # breakpoint()

    # create logger for saving testing results
    if args.tags and args.tl:
        log_path = config['log_dir']+'/TL/'+ args.tags + '/training/logs/training.log'
    elif args.tags:
        log_path = config['log_dir']+'/'+ args.tags + '/training/logs/training.log'
    else:
        log_path = config['log_dir'] + '/' + model_name + '/dfb_' + args.test_dataset[0] + '/training.log'

    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True) # create the directory if it does not exist
    logger = create_logger(log_path)
    logger.info('Experiment {}'.format(args.tags))
    logger.info('Save log to {}'.format(log_path))
    
    print("Save log to: ", log_path)
    # breakpoint()

    config['ddp']= args.ddp
    print("ddp: ", config['ddp'])

    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)
    generator = torch.Generator().manual_seed(config['manualSeed'])

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))


    # ----------------------------------------------- #
    if args.resume is not None:
        print("Resuming training from checkpoint: ", args.resume)

    ucf = False
    # split the training dataset into trn and val (80, 20 split), use the random seed in the config file
    # use random split
    # train_dataset = config['train_dataset']
    # load the dataset from config file
    if model_name == 'ucf':
        print("ucf - pairDataset")
        train_dataset = pairDataset(config = config, mode = 'train') # len 9000 -> but 9000 fake and 3000 real
        ucf = True
    else: 
        print("Xception - DeepfakeAbstractDataset")
        train_dataset = DeepfakeAbstractBaseDataset(
                config=config,
                mode='train',
            )
        # len 12000


    # print("type(train_dataset):", train_dataset)
    # print("loaded train dataset")
    
    # print("len(train_dataset): ", len(train_dataset)) # 12000
    # print("type(train_dataset): ", type(train_dataset)) 

    # print("train_dataset.data_dict[keys]: ", train_dataset.data_dict.keys())
    # print("len(train_dataset.data_dict['image']): ", len(train_dataset.data_dict['image']))
    # print("len(train_dataset.data_dict['label']): ", len(train_dataset.data_dict['label']))

    # if ucf:
    #     # use train_test_split to split the dataset into training and validation sets
    #     trn_dataset, vld_dataset = train_test_split(train_dataset, test_size=0.2, random_state= config['manualSeed'])
    #     trn_dataset = Subset(trn_dataset, trn_dataset.indices)
    #     vld_dataset = Subset(vld_dataset, vld_dataset.indices)

    # else: 
    trn_size = int(len(train_dataset) * 0.8)
    val_size = int(len(train_dataset)) - trn_size

    # print(f"dataset size: {len(train_dataset)}") # 12000
    # 600 imgs per challenge
    # 4 methods
    # 5 users
    # tot 12000 imgs
    # print(f"Training dataset size: {trn_size}") # 9600
    # print(f"Validation dataset size: {val_size}") # 2400

    trn_dataset, vld_dataset = torch.utils.data.random_split(train_dataset, [trn_size, val_size], generator)

    # return two datasets: trn_dataset and val_dataset
    print(f"Training dataset size: {len(trn_dataset)}") # 9600
    print(f"Validation dataset size: {len(vld_dataset)}") # 2400
    # print(f"Training dataset.dataset size: {len(trn_dataset.dataset)}") # 9600
    # print(f"Validation dataset.dataset size: {len(vld_dataset.dataset)}") # 2400
    print("type(trn_dataset)", type(trn_dataset)) # <class 'torch.utils.data.dataset.Subset'>
    print("type(vld_dataset)", type(vld_dataset)) # <class 'torch.utils.data.dataset.Subset'>
    print("type(trn_dataset.dataset)", type(trn_dataset.dataset)) # <class 'torch.utils.data.dataset.Subset'>
    print("type(vld_dataset.dataset)", type(vld_dataset.dataset)) # <class 'torch.utils.data.dataset.Subset'>
    # breakpoint()

    # trn_dataset = CustomSubset(trn_dataset, trn_dataset.indices)
    # vld_dataset = CustomSubset(vld_dataset, vld_dataset.indices)
    
    if model_name == 'ucf':
        # ucf = True
        trn_dataset = pairDataset(
                    config=config,
                    mode='train',
                    # indicies=trn_indices
                    indices=trn_dataset.indices
                )
        # len: 7200

        vld_dataset = pairDataset(
                    config=config,
                    mode='train',
                    # indicies=val_indices,
                    indices= vld_dataset.indices
                    # val = True
                )
    else: 
        trn_dataset = DeepfakeAbstractBaseDataset(
                        config=config,
                        mode='train',
                        # indicies=trn_indices
                        indicies=trn_dataset.indices
                    )

        vld_dataset = DeepfakeAbstractBaseDataset(
                        config=config,
                        mode='train',
                        # indicies=val_indices,
                        indicies= vld_dataset.indices
                        # val = True
                    )

    # print("check trn and vld splits after int the datasets class")
    print("len(trn_dataset)", len(trn_dataset)) # 9600
    print("len(vld_dataset)", len(vld_dataset)) # 2400
    print("type(trn_dataset)", type(trn_dataset)) # <class 'torch.utils.data.dataset.Subset'>
    print("type(vld_dataset)", type(vld_dataset)) # <class 'torch.utils.data.dataset.Subset'>
    # breakpoint()

    print("Defining the dataloaders")
    train_data_loader = \
            torch.utils.data.DataLoader(
                # dataset=train_subset,
                dataset = trn_dataset,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=  trn_dataset.collate_fn if model_name == 'ucf' else train_dataset.collate_fn,
                # collate_fn = trn_dataset.collate_fn
                # drop_last = True, # drop the last batch if it is not full
                )
    # # print(train_data_loader.keys())
    # # breakpoint()

    # # prepare the testing data loader
    # val_data_loader = prepare_val_dataloader(config, val_indices)

    val_data_loaders = {}

    # val_data_loader = prepare_test_dataloader(val_dataset, config, config['test_dataset'][0]) # only one test dataset is used 
    # # test_data_loaders = prepare_testing_data(config)
    # # test_data_loaders = prepare_testing_data(val_dataset, config)
    val_data_loader = torch.utils.data.DataLoader(
                # dataset=val_subset,
                dataset = vld_dataset,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn = vld_dataset.collate_fn if model_name == 'ucf' else train_dataset.collate_fn,
                # collate_fn = vld_dataset.collate_fn
                # drop_last = (test_name=='DeepFakeDetection'),
            )
    
    test_name = config['train_dataset'][0]
    print("test_name:", test_name)
    val_data_loaders[test_name] = val_data_loader 
    # print(val_data_loader.keys())

    # ---------------------------------------- #
    # # check data dict length
    trn_data_dict = train_data_loader.dataset.data_dict['image']
    print("len(trn_data_dict['image']): ", len(trn_data_dict)) # 9600
    val_data_dict = val_data_loaders[test_name].dataset.data_dict['image']
    print("len(val_data_dict['image']): ", len(val_data_dict)) # 2400

    trn_data_dict = train_data_loader.dataset.data_dict['label']
    print("len(trn_data_dict['label']): ", len(trn_data_dict)) # 9600
    val_data_dict = val_data_loaders[test_name].dataset.data_dict['label']
    print("len(val_data_dict['label']): ", len(val_data_dict)) # 2400

    trn_data_loader_keys = train_data_loader.dataset.data_dict.keys()
    print("trn_data_loader_keys: ", trn_data_loader_keys)
    vld_data_loader_keys = val_data_loaders[test_name].dataset.data_dict.keys()
    print("vld_data_loader_keys: ", vld_data_loader_keys)

    print("len(trn_dataloader):", len(train_data_loader))
    print("len(val_data_loader): ", len(val_data_loader))
    print("len(val_data_loaders[test_name]): ", len(val_data_loaders[test_name]))

    # breakpoint()

    
    
    # trn_indices = trn_dataset.indices
    # vld_indices = vld_dataset.indices

    # trn_data_dict = [train_dataset.data_dict['image'][i] for i in trn_indices]
    # print("len(trn_data_dict): ", len(trn_data_dict))  # 9600
    # val_data_dict = [train_dataset.data_dict['image'][i] for i in vld_indices]
    # print("len(val_data_dict): ", len(val_data_dict))  # 2400

    # trn_data_dict = [train_dataset.data_dict['label'][i] for i in trn_indices]
    # print("len(trn_data_dict): ", len(trn_data_dict))  # 9600
    # val_data_dict = [train_dataset.data_dict['label'][i] for i in vld_indices]
    # print("len(val_data_dict): ", len(val_data_dict))  # 2400   

    # breakpoint()
    
    # print("trn_data_dict['image']:", trn_data_dict)
    # print("val_data_dict['image']:", val_data_dict)

    # breakpoint()
    # ---------------------------------------- #

    # prepare the model (detector)
    print("prepare the model:", config['model_name'])
    
    model_class = DETECTOR[config['model_name']]
    model = model_class(config, args.tl) # initialize the model -> calls method build_backbone -> loads pretrained model 
    print("model.tl:", model.tl)
    print("model loaded!")
    
    # check_model_gradient(model)
    # breakpoint()

    # prepare the optimizer
    # if model.tl: 
    #     print("Using TL for training the model")
    #     optimizer = choose_optimizer(model.backbone.last_layer, config)
    # else:
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, args.tags, args.tl) 

    # breakpoint()

    # start training
    print("Start training...")
    for epoch in range(config['start_epoch'], config['nEpochs']): # + 1): # range(0, 10) -> 0, 1, 2, ..., 9 (10 is not included)
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
                    epoch=epoch,
                    train_data_loader=train_data_loader,
                    test_data_loaders=val_data_loaders,
                    # ucf = ucf,
                    gotcha = gotcha
                )
        if best_metric is not None:
            logger.info(f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric))) 
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values(): 
        writer.close()



if __name__ == '__main__':
    main()