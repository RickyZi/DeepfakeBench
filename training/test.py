"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder #RecorderFc
from collections import defaultdict

import argparse
from logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
# parser.add_argument('--detector_path', type=str, 
#                     default= './config/detector/xception.yaml',
#                     # '/home/zhiyuanyan/DeepfakeBench/training/config/detector/resnet34.yaml',
#                     help='path to detector YAML file')
parser.add_argument('--detector', type=str, default = 'xception', help='detector name')
parser.add_argument("--test_dataset", nargs="+") # define the test dataset name in the command line (more than one)
parser.add_argument('--tags', type=str, default="occlusion", help='tags for the test')
# parser.add_argument('--weights_path', type=str, 
#                     default='../dfb_weights/xception_best.pth'
#                     )
                    #'/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
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
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders, logger, model_name, dataset_name, tags):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict # get the data dictionary for the current dataset
        # print("data_dict['image']", data_dict['image'][0])
        print("data dict keys:", data_dict.keys())
        print("len(data_dict['image'])", len(data_dict['image']))
        for i in range(10):
            print(f"data_dict['image'][{i}]: {data_dict['image'][i]}")
        # compute loss for each dataset
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])

        # predictions_nps: numpy array of predictions
        # print the model predictions to check if the model has one or more outputs

        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'], model = model_name, dataset = dataset_name, tags = tags)
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}") # print the dataset name 
        logger.info(f"dataset: {key}") # save the info for each dataset to a log file
        for k, v in metric_one_dataset.items():
            
            if k == 'precision' or k == 'recall' or k == 'img_path_collection' or k == 'label' or k == 'pred':
                # tqdm.write(f"{k}: {v[100]}") # print only the first 100 values
                logger.info(f"{k}: {list(v)}")
                # tqdm.write(f"{k}: {v[:50]}") # print only the last 50 values
            else:
                tqdm.write(f"{k}: {v:.4f}") # print the metric value for each dataset
                logger.info(f"{k}: {v:.4f}") # save the metric value for each dataset to a log file

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # Overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # Load the new state dict
    model.load_state_dict(model_dict)

def print_model_state_dict(model):
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        print(f"Key: {key}, Value: {value.shape}")



# class ModifiedModel(nn.Module):
#     def __init__(self, original_model, num_classes):
#         super(ModifiedModel, self).__init__()
#         self.original_model = original_model
        
#         # Modify the layer to match the checkpoint dimensions
#         self.original_model.head_spe.mlp[2] = nn.Linear(512, num_classes)  # Adjust the output dimensions to 5

#     def forward(self, x):
#         return self.original_model(x)



def main():

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
    # parse the detector config
    with open(detector_yaml, 'r') as f:
        config = yaml.safe_load(f)
    # parse the test config
    with open('./config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)

    # since in the rest of the code they use only config, we need to update it with the test config
    config.update(config2) # update the config with the test config info -> missing??

    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']

    # weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        dataset_name = args.test_dataset[0]
        config['test_dataset'] = args.test_dataset
        
    else:
        dataset_name = config['test_dataset'][0]
    
    
    # if args.weights_path:
    #     config['weights_path'] = args.weights_path
    #     weights_path = args.weights_path
    if weights_path:
        config['weights_path'] = weights_path
        
    # print("dataset_name", dataset_name)
    # breakpoint()
    

    # create logger for saving testing results
    if args.tags:
        log_path = config['log_dir']+'/'+ args.tags + '/testing/logs/test_output.log'
    else:
        log_path = config['log_dir'] + '/' + model_name + '/dfb_' + dataset_name + '/test_output.log'

    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True) # create the directory if it does not exist
    logger = create_logger(log_path)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)


    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    # model = ModifiedModel(model, 2).to(device)
    
    # print(model)
    # print_model_state_dict(model)

    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0

        # print("epoch:", epoch)

        ckpt = torch.load(weights_path, map_location=device)
        # model.load_state_dict(ckpt, strict=True)
        # model.load_state_dict(ckpt, strict=False)


        # Modify the state_dict to match the current model's dimensions
        # state_dict = ckpt['state_dict']
        # Inspect the keys in the checkpoint
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt  # Assume the checkpoint itself is the state_dict

        model_state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            if name in model_state_dict:
                if param.shape != model_state_dict[name].shape:
                    print(f"Skipping loading parameter {name}, required shape {model_state_dict[name].shape}, loaded shape {param.shape}")
                    state_dict[name] = model_state_dict[name]
        
        model.load_state_dict(state_dict, strict=False) # False to allow for more flexibility when loading slightly modified architecture

        print('===> Load checkpoint done!')

        # print(model)
        # print_model_state_dict(model)
        # breakpoint()

    else:
        print('Fail to load the pre-trained weights')
    

    # breakpoint()
    # exit()

    # start testing
    best_metric = test_epoch(model, test_data_loaders, logger, model_name, dataset_name, args.tags)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
