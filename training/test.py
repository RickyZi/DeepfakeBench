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
from metrics.utils import get_test_metrics, gotcha_test_metrics
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

# from trainer.trainer import Trainer
from detectors import DETECTOR
# from metrics.base_metrics_class import Recorder #RecorderFc
# from collections import defaultdict

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
parser.add_argument('--tl', action = "store_true", default=False)
parser.add_argument('--gen', action = "store_true", default=False)
# parser.add_argument('--weights_path', type=str, 
#                     default='../dfb_weights/xception_best.pth'
#                     )
                    #'/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw/cnn_aug/resnet34_2023-05-20-16-57-22/test/FaceForensics++/ckpt_epoch_9_best.pth')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

# --------------------------------------------------------------------------------------------- #
# python test.py --detector "xception" --test-dataset "no_occlusio" --tags "Xception-no-occ-GEN"
# python test.py --detector "xception" --tags "Xception_dfb_occ_TL" --tl 
# python test.py --detector "xception" --tags "Xception_dfb_no_occ_TL" --tl --test_dataset "no_occlusion"
# --------------------------------------------------------------------------------------------- #

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

        # print("predictions", predictions)
        # print("type(predictions)", type(predictions))
        # print("label_lists[:10]", label_lists[:10])
        # print("prediction_lists[:10]", prediction_lists[:10])
        # print("feature_lists[:10]", feature_lists[:10])
        
        # breakpoint()
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders, logger,  tags, gotcha = False): # model_name, dataset_name,
    # set model to eval mode
    model.eval()
    print("test_epoch gotcha: ", gotcha)
    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict # get the data dictionary for the current dataset
        # print("data_dict['image']", data_dict['image'][0])
        # print("data dict keys:", data_dict.keys())
        # print("len(data_dict['image'])", len(data_dict['image']))
        # for i in range(10):
        #     print(f"data_dict['image'][{i}]: {data_dict['image'][i]}")
        # compute loss for each dataset
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])

        # predictions_nps: numpy array of predictions
        # print the model predictions to check if the model has one or more outputs

        
        # compute metric for each dataset
        if gotcha:
            metric_one_dataset = gotcha_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'], tags = tags, tl = args.tl, gen = args.gen)
        else:
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=data_dict['image'], tags = tags, tl = args.tl, gen = args.gen) # model = model_name, dataset = dataset_name
        metrics_all_datasets[key] = metric_one_dataset
        
        # log the experiment info
        logger.info(f"test: {tags}")

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

    print("tl: ", args.tl)
    print("dataset: ", args.test_dataset)
    print("dataset[0]:", args.test_dataset[0])
    # breakpoint()

    gotcha = False 

    if args.detector == 'xception':
        detector_yaml = './config/detector/xception.yaml'       
        model_name = 'xception'
        if args.tl and args.test_dataset[0] == "occlusion":
            # if args.pretrained:
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL/xception_2024-09-20-13-28-49/test/occlusion/ckpt_best.pth' # focal_loss [USE THIS!!!!]
            # '/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_def_frame_num/xception_2024-10-15-13-23-43/test/occlusion/ckpt_best.pth' # test with 32 frames for trn
            #
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_originalTrain/xception_2024-10-15-09-03-15/test/occlusion/ckpt_best.pth' # original DFB training strategy

            # ---------------------------------------- #
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_adjChannel/xception_2024-10-14-12-51-27/test/occlusion/ckpt_best.pth' # adjust_channel -> RESULTS IMPROVE!!!!!
            # ---------------------------------------- #
            # '/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_lrSched/xception_2024-10-14-12-30-11/test/occlusion/ckpt_best.pth' # linear lr_sched
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_test_dropout_and_lr/xception_2024-10-14-12-09-28/test/occlusion/ckpt_best.pth' # cosine_lr_sched + dropout (0.3)
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_test_dropout/xception_2024-10-11-13-35-39/test/occlusion/ckpt_best.pth' # dropout = 0.2
            
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_test_dropout_and_lr/xception_2024-10-11-14-01-02/test/occlusion/ckpt_best.pth' # test dropout + lr_scheduler

            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL_test/xception_2024-10-10-13-39-04/test/occlusion/ckpt_best.pth' #test auc as vld metric
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL/xception_2024-10-10-13-21-53/test/occlusion/ckpt_best.pth' # test new validation definitio
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_occ_TL/xception_2024-09-20-13-28-49/test/occlusion/ckpt_best.pth' # focal_Loss [use this!!!!]
            #'/home/rz/DeepfakeBench/training/results/Xception-dfb-occ-TL/xception_2024-09-11-11-41-36/test/occlusion/ckpt_best.pth'
            # weights_path = '/home/rz/DeepfakeBench/training/results/Xception-dfb-occ-TL/xception_2024-09-17-12-32-52/test/occlusion/ckpt_best.pth'
            print(f"using TL {model_name} model: {weights_path}")
            # config['pretrained'] = weights_path
            # print(config['pretrained'])
        elif args.tl and args.test_dataset[0] == "no_occlusion":
            # if args.pretrained:
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_no_occ_TL/xception_2024-10-15-09-55-04/test/no_occlusion/ckpt_best.pth' # focal_loss [USE THIS!!!!]
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_no_occ_TL/xception_2024-09-20-14-04-26/test/no_occlusion/ckpt_best.pth' #USE_THIS
            # '/home/rz/DeepfakeBench/training/results/Xception-dfb-occ-TL-focalLoss/xception_2024-09-18-08-35-37/test/occlusion/ckpt_best.pth'
            #'/home/rz/DeepfakeBench/training/results/Xception-dfb-occ-TL/xception_2024-09-11-11-41-36/test/occlusion/ckpt_best.pth'
            # weights_path = '/home/rz/DeepfakeBench/training/results/Xception-dfb-occ-TL/xception_2024-09-17-12-32-52/test/occlusion/ckpt_best.pth'
            print(f"using TL {model_name} model: {weights_path}")
            # config['pretrained'] = weights_path
            # print(config['pretrained'])
        elif args.tl and args.test_dataset[0] == "gotcha_occlusion":
            # model trained with 100 imgs per class
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/Xception_gotcha_occ_TL_def_frame_num/xception_2024-10-15-12-40-10/test/gotcha_occlusion/ckpt_best.pth' # test 32 frames for trn/tst (default frame_num)'
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_gotcha_occ_TL/xception_2024-10-10-09-58-48/test/gotcha_occlusion/ckpt_best.pth' # use this!!
            #
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_gotcha_occ_TL/xception_2024-10-10-09-58-48/test/gotcha_occlusion/ckpt_best.pth' # use this!!
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_gotcha_occ_TL/xception_2024-10-03-12-21-24/test/gotcha_occlusion/ckpt_best.pth'
            print(f"using TL {model_name} model: {weights_path}")
            gotcha = True

        elif args.tl and args.test_dataset[0] == "gotcha_no_occlusion":
            # model trained with 200 imgs per class
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/Xception_gotcha_no_occ_TL/xception_2024-10-15-13-57-27/test/gotcha_no_occlusion/ckpt_best.pth' # test 32 frames for trn/tst
            #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_gotcha_no_occ_TL/xception_2024-10-03-13-13-29/test/gotcha_no_occlusion/ckpt_best.pth'
            print(f"using TL {model_name} model: {weights_path}")
            gotcha = True
            
        else:
            weights_path = './pretrained/xception_best.pth'
            print("using default pretrained model: ", weights_path)
            if args.test_dataset[0] == 'gotcha_occlusion' or args.test_dataset[0] == 'gotcha_no_occlusion':
                gotcha = True

        
    elif args.detector == 'ucf':
        detector_yaml = './config/detector/ucf.yaml'
        model_name = 'ucf'
        # load pretrained model
        if args.tl and args.test_dataset[0] == 'occlusion':
            # if args.pretrained:
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL/ucf_2024-10-16-11-27-39/test/occlusion/ckpt_best.pth' # base UCF training (100 ex in trn/tst and focal_loss)
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL_lr_sched/ucf_2024-09-30-09-16-49/test/occlusion/ckpt_best.pth' # ucf w/ lr_scheduler 10 epochs lr_step = 2
            # '/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL_lr_sched/ucf_2024-09-30-08-09-04/test/occlusion/ckpt_best.pth'ucf w/ lr_Scheduler 8 epochs - lr_step = 2
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL_lr_sched/ucf_2024-09-26-15-06-25/test/occlusion/ckpt_best.pth' # ucf w/ lr_Scheduler 5 epochs - lr_step = 1
            # weights_path = '/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL_focal_loss/ucf_2024-09-26-12-48-10/test/occlusion/ckpt_best.pth'
            # ---------------------------------------------------------------------------------------------------------------- #
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_occ_TL/ucf_2024-09-26-08-58-01/test/occlusion/ckpt_best.pth'
            #'/home/rz/DeepfakeBench/training/results/UCF-dfb-occ-TL/ucf_2024-09-17-07-59-22/test/occlusion/ckpt_best.pth'
             # ---------------------------------------------------------------------------------------------------------------- #
            print(f"using TL {model_name} model: {weights_path}")
            # config['pretrained'] = weights_path
            # print(config['pretrained'])
        elif args.tl and args.test_dataset[0] == 'no_occlusion':
            # if args.pretrained:
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_no_occ_TL/ucf_2024-10-16-12-32-33/test/no_occlusion/ckpt_best.pth' # base UCF training with 100 ex per trn/tst and focal_loss 
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_no_occ_TL_lr_sched/ucf_2024-09-30-11-38-36/test/no_occlusion/ckpt_best.pth' # ucf w/step_lr_scheduler each 2 out of 10 training epochs
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_no_occ_TL_focal_loss/ucf_2024-09-26-13-35-50/test/no_occlusion/ckpt_best.pth'
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_dfb_no_occ_TL/ucf_2024-09-26-09-44-17/test/no_occlusion/ckpt_best.pth'
            print(f"using TL {model_name} model: {weights_path}")

        elif args.tl and args.test_dataset[0] == "gotcha_occlusion":
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_occ_TL/ucf_2024-10-16-08-10-31/test/gotcha_occlusion/ckpt_best.pth' # ucf 32 imgs in trn/test & auc test metric
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_occ_TL/ucf_2024-10-15-14-13-55/test/gotcha_occlusion/ckpt_best.pth' # ucf trained with 32 imgs per user but auc as test metric
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_occ_TL/ucf_2024-10-09-12-04-14/test/gotcha_occlusion/ckpt_best.pth' # ucf trained on the whole dataset
            #  
            print(f"using TL {model_name} model: {weights_path}")
            gotcha = True

        elif args.tl and args.test_dataset[0] == "gotcha_no_occlusion":
            weights_path = '/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-08-58-28/test/gotcha_no_occlusion/ckpt_best.pth'
            #'/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-07-29-26/test/gotcha_no_occlusion/ckpt_best.pth' # ucf trained with 32 imgs but auc as test metric
            print(f"using TL {model_name} model: {weights_path}")
            gotcha = True

        else:
            weights_path = './pretrained/ucf_best.pth'
            print(f"using default pretrained {model_name} model: {weights_path}")
            if args.test_dataset[0] == 'gotcha_occlusion' or args.test_dataset[0] == 'gotcha_no_occlusion':
                gotcha = True
    else:
        raise NotImplementedError('detector {} is not implemented'.format(args.detector))

    print("gotcha: ", gotcha)
    # breakpoint()

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
        config['test_dataset'] = args.test_dataset
    else:
        args.test_dataset = config['test_dataset']
    
    print("dataset:")
    print(args.test_dataset)
    print(config['test_dataset'])

    # if args.test_dataset == ['gotcha_occlusion'] or args.test_dataset == ['gotcha_no_occlusion']:
    #     gotcha = True
    #     # # config['frame_num']['train'] = 900
    #     # config['frame_num']['test'] = 160 #900
    #     config['test_batchSize'] = 128 # or 128?
    
    # print("config['frame_num']['train']", config['frame_num']['train'])
    print("config['frame_num']['test']", config['frame_num']['test'])
    print("config['test_batchSize']: ", config['test_batchSize'])
    # breakpoint() 
    # if args.weights_path:
    #     config['weights_path'] = args.weights_path
    #     weights_path = args.weights_path
    if weights_path:
        config['weights_path'] = weights_path
        
    # print("dataset_name", dataset_name)
    # breakpoint()
    
    if config['backbone_config']['dropout'] != False:
        print("deactivating dropout for testing!")
        config['backbone_config']['dropout'] = False
    else:
        print("dropout: False")

    # create logger for saving testing results
    if args.tags and args.tl:
        log_path = config['log_dir']+'/TL/'+ args.tags + '/testing/logs/test_output.log'
    elif args.tags and args.gen:
        log_path = config['log_dir']+'/GEN/'+ args.tags + '/testing/logs/test_output.log'
    elif args.tags:
        log_path = config['log_dir']+'/'+ args.tags + '/testing/logs/test_output.log'
    else:
        log_path = config['log_dir'] + '/' + model_name + '/dfb_' + args.test_dataset + '/test_output.log'

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

    # add weights_path to log file
    logger.info(f"model weights path: {weights_path}")

    # model = ModifiedModel(model, 2).to(device)
    
    # print(model)
    # breakpoint()
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
    best_metric = test_epoch(model, test_data_loaders, logger, args.tags, gotcha) # model_namedataset_name, args.tags)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
