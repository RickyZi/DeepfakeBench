import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
# from torchvision.models import mobilenet_v2
import os

# import cv2

import argparse # for command line arguments
# pip install efficientnet_pytorch # need to install this package to use EfficientNet
# from efficientnet_pytorch import EfficientNet

from logger import * # import the logger functions
import timm # for using the XceptionNet model (pretrained)
# pip install timm # install the timm package to use the XceptionNet model
from papers_data_augmentations import *

import fornet
from fornet import *

import yaml

from dfb import *

from dfb.dfb_detectors import DETECTOR


# -------------------------------------------------------------------------------- #
# How to run the script with the arguments
# python main_test.py --model "MobileNetV2" --save-model-path "./model/best_model.pth" --dataset "thesis_occlusion" --wandb --tags "TestRun" --training True 
# use the MobileNetV2 model to train the model on the thesis occlusion dataset and save it at the specified path, add the tags to the wandb session
# wandb is used to log the results of the model training
# tags are used to find the wandb session online and to log the results of the model training/testing, to save the model in a folder with the tags, 
# to save the log files, and also to save the AUROC and AUPRC plots
# to test the model use the same command but remove the --training flag
# the other arguments will be taken from the default values in the parser
# ------------------------------------------------------------------------------------------- #
# MobileNetV2 - command to run trained mobilenetv2 (default model) model and test metrics and roc/aupr curves
# python main_test_v2.py --save-model-path "./model/best_focal_loss_model/new_focal_loss_no_resize_rotation_color_jitter.pth"  --tags "MobileNetV2-best-test-metrics" --save-log --training
# note copied best focal loss model in the folder ./model/MobileNetV2_baseline_occ/best_model.pth
# python main_test_v2.py --save-model-path "./model/MobileNetV2_baseline_no_occ/best_model.pth" --dataset "thesis_no_occ"  --tags "MobileNetV2-baseline-no-occlusion" --save-log --wandb --training

# EfficientNetB4
# python main_test_v2.py --model "efficientnetb4" --save-model-path "./model/EfficientNet_B4_baseline_occ/best_model.pth" --tags "EfficientNet_B4-baseline-occlusion" --save-log --wandb --training
# python main_test_v2.py --model "efficientnetb4" --save-model-path "./model/EfficientNet_B4_baseline_no_occ/best_model.pth" --dataset "thesis_no_occ" --tags "EfficientNet_B4-baseline-no-occlusion" --save-log --wandb --training

# XceptionNet
# python main_test_v2.py --model "xception" --save-model-path "./model/XceptionNet_baseline_occ/best_model.pth" --tags "XceptionNet-baseline-occlusion" --save-log --wandb --training
# python main_test_v2.py --model "xception" --save-model-path "./model/XceptionNet_baseline_no_occ/best_model.pth" --dataset "thesis_no_occ" --tags "XceptionNet-baseline-no-occlusion" --save-log --wandb --training


# ----------------------------------------------- #
# Milan papers
# ----------------------------------------------- #
# EfficetNetB4_FF - 
# thesis-occ-dataset
# python main_test_v2.py --model "efficientnetb4_ff" --data-aug "milan" --tags "EfficientNetB4_FF_occ_focal_loss" --save-log --wandb 
# python main_test_v2.py --model "efficientnetb4_ff" --loss "bce" --data-aug "milan" --tags "EfficientNetB4_FF_occ_bce_loss" --save-log --wandb 

# thesis-no-occ
# python main_test_v2.py --model "efficientnetb4_ff" --data-aug "milan" --dataset "thesis_no_occ" --tags "EfficientNetB4_FF_no_occ_focal_loss" --save-log --wandb 
# python main_test_v2.py --model "efficientnetb4_ff" --loss "bce" --data-aug "milan" --dataset "thesis_no_occ" --tags "EfficientNet_B4_FF_no_occ_bce_loss" --save-log --wandb 

# efficentnetb4_dfdc
# thesis-occ-dataset
# python main_test_v2.py --model "efficientnetb4_dfdc" --data-aug "milan" --tags "EfficientNetB4_DFDC_occ_focal_loss" --save-log --wandb 
# python main_test_v2.py --model "efficientnetb4_dfdc" --loss "bce" --data-aug "milan" --tags "EfficientNetB4_DFDC_occ_bce_loss" --save-log --wandb 

# thesis-no-occ
# python main_test_v2.py --model "efficientnetb4_dfdc" --data-aug "milan" --dataset "thesis_no_occ" --tags "EfficientNetB4_DFDC_no_occ_focal_loss" --save-log --wandb 
# python main_test_v2.py --model "efficientnetb4_dfdc" --loss "bce" --data-aug "milan" --dataset "thesis_no_occ" --tags "EfficientNetB4_DFDC_no_occ_bce_loss" --save-log --wandb 

# ----------------------------------------------- #
# DFB papers
# check what we need to test the model accordign to their train/test definition
# ----------------------------------------------- #
# EfficientNetB4

# XceptionNet
# python test.py --model "xception_dfb" --dataset "dfb_no_occ" --data-aug "dfb" --save-logs --wandb
# UCF


# ------------------------------------------------------------------------------------------- #
# Create the argument parser
def get_args_parse():
    parser = argparse.ArgumentParser(description='Model Training and Testing')
    # Add arguments

    # model parameters
    parser.add_argument('--model', type=str, default='mobilenetv2', help='Model to use for training and testing')

    
    return parser


# ----------------------------------------- #
# main function to train and test the model #
# ----------------------------------------- #


    print("Not using wandb for logging")


# ---------------------------------------------------------- #
# @TODO: try to add another model to test (i.e. EfficientNetB4)
# need to update the model loading part, also need to update the training and testing transformations?
# python main_upd.py --model efficientnetb4 --save-model-path './best_model/efficentnet_test/best_model.pth' --training 
# ---------------------------------------------------------- #
# Load the pre-trained model (MobileNetV2 or EfficientNetB4) #
# ---------------------------------------------------------- #   

# if args.scratch and args.model == 'mobilenetv2':
# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
# ----------------------- #
# --- BASELINE MODELS --- #
# ----------------------- #

# print(args.save_model_path)

if args.model == 'mobilenetv2': 
    print("Loading MobileNetV2 model")
    # model = mobilenet_v2(pretrained=True)
    model = models.mobilenet_v2(weights = 'MobileNet_V2_Weights.IMAGENET1K_V2') 
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier layer
    model.classifier[1] = nn.Linear(model.last_channel, 1) # only 1 output -> prob of real of swap face

    model_name = 'MobileNetV2' # add the model name to the model object


    if args.save_model_path is None:
        print("using pretrained ImageNet model")
        
    else:
        print("loading pretrained model")
        # load the pre-trained model for testing (load model weights)
        pretrained_model_path = args.save_model_path 
        print("model saved in:", pretrained_model_path)
        model.load_state_dict(torch.load(pretrained_model_path)) 

    model.to(device)
    print("Model loaded!")
    print(model)
    exit()

elif args.model == 'efficientnetb4':
    print("Loading EfficientNetB4 model")
    # pip install efficientnet_pytorch
    # run this command to install the efficientnet model
    # model = EfficientNet.from_pretrained('efficientnet-b4')

    model = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1') # correct way to call pre-trained model
    # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier layer
    # model._fc = nn.Linear(model._fc.in_features, 1) # only 1 output -> prob of real of swap face
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1) # modify the last layer of the classifier to have 1 output -> prob of real of swap face
    # last layer of EfficientNetB4 is a Linear layer (classifier) with 1000 outputs (for ImageNet) -> change it to 1 output
    
    model_name = 'EfficientNet_B4' # add the model name to the model object

    # load the pre-trained model for testing (load model weights)
    pretrained_model_path = args.save_model_path 
    print("model saved in:", pretrained_model_path)
    model.load_state_dict(torch.load(pretrained_model_path)) 
    model.to(device)
    print("Model loaded!")

    # for key in model.keys():
    #     print(key)

    print(model)


    exit()
    
elif args.model == 'xception':
    print("Loading pretrained XceptionNet model...")
    # load the xceptionet model
    # pip install timm
    # import timm
    model = timm.create_model('xception', pretrained=True, num_classes=1) # only 1 output -> prob of real of swap face
    model_name = 'XceptionNet' # add the model name to the model object

    # load the pre-trained model for testing (load model weights)
    pretrained_model_path = args.save_model_path 
    print("model saved in:", pretrained_model_path)
    model.load_state_dict(torch.load(pretrained_model_path)) 
    model.to(device)

    print("Model loaded!")

# ---------------------- #
# ---- MILAN MODELS ---- #
# ---------------------- #

# add the other models to test here
elif args.model == 'efficientnetb4_ff':
    print("Loading EfficientNetB4_FF model")
    args.save_model_path = './_pretrained_models_papers_/icpr2020dfdc_weights/EfficientNetB4_FFPP/bestval.pth'
    model_name = "EfficientNetB4_FF"

    model_state = torch.load(args.save_model_path, map_location = "cpu")
    # net = "EfficientNetB4"
    # model.load_state_dict(model_state)
    net_name = "EfficientNetB4"
    net_class = getattr(fornet, net_name)
    # net: FeatureExtractor = net_class().eval().to(device)
    model: FeatureExtractor = net_class().to(device)
    incomp_keys = model.load_state_dict(model_state['net'], strict=True)
    print(incomp_keys)
    # print(model)
    print('Model loaded!')

    # # state_dict = torch.load(args.save_model_path)
    # for name, weights in model_state.items():
    #     print("name:", name)
    #     # print("weights:", weights)


    # print(model_state['net'])

    # print(model)

    # for key in model_state.keys():
    #     print(key)


    # exit()

    # use net in the rest of the code and not model

    # model = net.load_state_dict(model_state['net'], strict = True)
    # print(model)
    # print('Model loaded!')

    # https://github.com/polimi-ispl/icpr2020dfdc/blob/bbd64115e612e50416fb64fa8f60393fe4642dc0/test_model.py#L93
    #  # Load net
    # net_class = getattr(fornet, net_name)

    # # load model
    # print('Loading model...')
    # state_tmp = torch.load(model_path, map_location='cpu')
    # if 'net' not in state_tmp.keys():
    #     state = OrderedDict({'net': OrderedDict()})
    #     [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    # else:
    #     state = state_tmp
    # net: FeatureExtractor = net_class().eval().to(device)

    # incomp_keys = net.load_state_dict(state['net'], strict=True)
    # print(incomp_keys)
    # print('Model loaded!')
    
    
elif args.model == 'efficientnetb4_dfdc':
    print("Loading EfficientNetB4_DFDC model")
    args.save_model_path = './_pretrained_models_papers_/icpr2020dfdc_weights/EfficientNetB4_DFDC/bestval.pth'
    model_state = torch.load(args.save_model_path, map_location = "cpu")

    # net_name = "EfficientNetB4"
    net_class = getattr(fornet, "EfficientNetB4")
    # net: FeatureExtractor = net_class().eval().to(device)
    model: FeatureExtractor = net_class().to(device)
    incomp_keys = model.load_state_dict(model_state['net'], strict = True)
    model_name = "EfficientNetB4_DFDC"
    # incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!')

    # print(model)
    # exit()

    # incomp_keys cannot be loaded to device
    # python main_test_v2.py --model "efficentnetb4_dfdc" --loss "bce"  --dataset "thesis_no_occ"  --data-augmentation "milan"  --tags "EfficentNetB4_DFDC_no_occ_bce_loss" --save-log --wandb
    # optimizer = optim.Adam(model.parameters(), lr= args.lr) #, weight_decay= args.weight_decay) # add weight decay

    # AttributeError: '_IncompatibleKeys' object has no attribute 'parameters'

# ---------------------- #
# ----- DFB MODELS ----- #
# ---------------------- #

# NOT SURE ABOUT DFB MODEL CHECKPOINTS -> PROBABLY NEED TO ADD ALSO THE MODELS DEFINITION FROM THE REPO
# ALSO PROBABLY NEED TO USE A CONFIG FILE AS THEY DO IN THE DFB REPO

elif args.model == "xception_dfb":
    print("Loading XceptionNet_DFB model")
    args.save_model_path = './_pretrained_models_papers_/deepfakebench_weights/xception_best.pth'
    # model_state = torch.load(args.save_model_path, map_location = "cpu")
    # model.load_state_dict(model_state)
    model_name = "XceptionNet_DFB"

    detector_path = './dfb/dfb_config/xception.yaml'
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    print("read config file")
    # ...
    # prepare the model (detector)
    print("prepare the model")
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    print("checkpoint")
    ckpt = torch.load(args.save_model_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print("ckpt loaded")
    print(model)
    print(model.loss_func)
    print("Model loaded!")
    # model_state = torch.load(args.save_model_path, map_location = "cpu")
    # model.load_state_dict(model_state)

    # exit() 
    breakpoint()

elif args.model == "ucf_dfb":
    print("Loading UCF_DFB model")
    args.save_model_path = './_pretrained_models_papers_/deepfakebench_weights/ucf_best.pth'
    # model_state = torch.load(args.save_model_path, map_location = "cpu")
    # model.load_state_dict(model_state)
    model_name = "UCF_DFB"

    detector_path = './dfb/dfb_config/ucf.yaml'
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    print("read config file")
    # ...
    # prepare the model (detector)
    print("prepare the model")
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    print("checkpoint")
    ckpt = torch.load(args.save_model_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print("ckpt loaded")
    print(model)
    print(model.loss_func)
    print("Model loaded!")

    # exit()
    breakpoint()

else:
    print("Model not supported")
    exit()



# # Move the model to GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print("device", device) # output: cpu -> gpu drivers not active/updated??
# model = model.to(device)


# load the dataset -> for now only the face occlusion dataset (thesis), later add the GOTCHA dataset
# NOTE: also need to fix the customDataset.py to load the GOTCHA dataset as well!!
# dataset
gotcha = False
if args.dataset == 'thesis_occ':
    # train_dir = '/media/data/rz_dataset/users_face_occlusion/training/'
    test_dir = '/media/data/rz_dataset/users_face_occlusion/testing/'
    
elif args.dataset == 'thesis_no_occ':
    # train_dir = '/media/data/rz_dataset/user_faces_no_occlusion/training/'
    test_dir = '/media/data/rz_dataset/user_faces_no_occlusion/testing/'
    
elif args.dataset == 'milan_occ':
    test_dir = '/media/data/rz_dataset/milan_faces/occlusion/testing/'

elif args.dataset == 'milan_no_occ':
    test_dir = '/media/data/rz_dataset/milan_faces/no_occlusion/testing/'
    # testing -> user_300229  user_792539
    # training -> all the other users

elif args.dataset == 'dfb_occ':
    test_dir = '/media/data/rz_dataset/dfb_faces/occlusion/testing/'

elif args.dataset == 'dfb_no_occ':
    test_dir = '/media/data/rz_dataset/dfb_faces/occlusion/testing/'

# GOTCHA dataset -> might need to modify the dataloader since gotcha has a slightly different structure than the other datasets
# need to check also the gotcha dataset -> divide images into occlusion and no occlusion
# elif args.dataset == 'gotcha_occ':
#     # FIX PATH!!!
#     # train_dir = '/media/data/rz_dataset/gotcha/training/'
#     test_dir = '/media/data/rz_dataset/gotcha/testing/'
#     gotcha = True

# elif args.dataset == 'gotcha_no_occ':
#     # FIX PATH!!!
#     # train_dir = '/media/data/rz_dataset/gotcha/training'
#     test_dir = '/media/data/rz_dataset/gotcha/testing'
#     gotcha = True

# @TODO: add the GOTCHA dataset here and in the customDataset.py for the dataloader

else:
    print("Dataset not supported")
    exit()

print("dataset:", args.dataset)
# print("train_dir:", train_dir)
# print("test_dir:", test_dir)



# num_epochs = args.num_epochs
# # lr = args.lr # 0.001 = 1e-3 (default value)
# # wheight_decay = args.weight_decay # 1e-5 (default value)
# # patience = args.patience # 3 (number of epochs to wait for improvement before stopping)
# best_val_loss = float('inf')  # Initialize best validation loss
# early_stopping_counter = 0  # Counter to keep track of non-improving epochs



# --------------------------------- #
# Define the dataset and dataloaders
# --------------------------------- #
# if args.model == 'mobilenetv2':

generator = torch.Generator().manual_seed(42) # fix generator for reproducible results (hopefully good)

batch_size = args.batch_size

if args.data_aug == 'default':
    print("Using default data augmentation (thesis)")
    # Transformations for training data
    # train_transform = transforms.Compose([
    #     # base trasnf
    #     # transforms.Resize((256,256)),# interpolation = transforms.InterpolationMode.BILINEAR), # might not need since randomresizedcrop extracts rand crop and then resize it to 224 dim
    #     # Resize(256) -> resize the image to 256*h/2, 256, in our case is the same since the images have the same width and height
    #     transforms.RandomResizedCrop((224,224)), #interpolation = BICUBIC), # extracts random crop from image (i.e. 300x300) and rescale it to 224x224
    #     transforms.RandomHorizontalFlip(), # helps the training
    #     # augmentations
    #     transforms.RandomRotation((-5,5)), # rotate the image by a random angle between -5 and 5 degrees
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # adjust the brightness, contrast, saturation and hue of the image
    #     # transforms.GaussianBlur((5,9), sigma=(0.1, 2.0)), # blur the image with a random kernel size and a random standard deviation
    #     # kernel size (5,9) taken from pytorch doc, for sigma used the default values -> https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html#sphx-glr-auto-examples-transforms-plot-transforms-illustrations-py
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # Transformations for testing data
    test_transform = transforms.Compose([
        transforms.Resize((256,256)), 
        # BICUBIC is used for EfficientNetB4 -> check the documentation
        # BICUBIC vs BILINEAR -> https://discuss.pytorch.org/t/what-is-the-difference-between-bilinear-and-bicubic-interpolation/20920 
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # # training and validation dataset
    # train_dataset = FaceImagesDataset(train_dir, train_transform)
    # train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training 
    # val_size = len(train_dataset) - train_size # 20% of the dataset for validation

    # # split the dataset into training and validation
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
    # # returns two datasets: train_dataset and val_dataset

    # -------------------------------------------------------------------------------- #
    # old dataloader definition
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # -------------------------------------------------------------------------------- #

    # # test with samplers for the dataloader
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)
    # sampler_val   = torch.utils.data.SequentialSampler(val_dataset)

    # # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    # train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train) # batch_sampler = batch_sampler_train) #
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)

    # test dataloader 
    test_dataset = FaceImagesDataset(test_dir, test_transform)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=sampler_test) # shuffle=False)

elif args.data_aug == 'milan':
    print("using data augmentation from Milan paper")
    # train_transform = milan_train_transf()
    test_transform = milan_test_transf()

    # # training and validation dataset
    # train_dataset = FaceImagesAlbu(train_dir, train_transform)
    # train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training 
    # val_size = len(train_dataset) - train_size # 20% of the dataset for validation

    # # split the dataset into training and validation
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
    # # test with samplers for the dataloader
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)
    # sampler_val   = torch.utils.data.SequentialSampler(val_dataset)
    # train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train) # batch_sampler = batch_sampler_train) #
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)

    # test dataloader 
    test_dataset = FaceImagesAlbu(test_dir, test_transform)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=sampler_test) # shuffle=False)

elif args.data_aug == 'dfb':
    print("using data augmentation from DFB paper")
    # train_transform = dfb_data_aug()
    test_transform = dfb_data_aug()
    # NOTE: as it seems from the repo they use the same data augmentations for both training and testing

    # # training and validation dataset
    # train_dataset = FaceImagesAlbu(train_dir, train_transform)
    # train_size = int(0.8 * len(train_dataset)) # 80% of the dataset for training 
    # val_size = len(train_dataset) - train_size # 20% of the dataset for validation

    # # split the dataset into training and validation
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator)
    # # test with samplers for the dataloader
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)
    # sampler_val   = torch.utils.data.SequentialSampler(val_dataset)
    # train_dataloader = DataLoader(train_dataset,batch_size=batch_size, sampler=sampler_train) # batch_sampler = batch_sampler_train) #
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler_val)

    # test dataloader 
    test_dataset = FaceImagesAlbu(test_dir, test_transform)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, batch_size=64, sampler=sampler_test) # shuffle=False)

else:
    print("Data augmentation not supported")
    exit()

# print("data augmentations: ", args.data_aug)



