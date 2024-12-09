import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import argparse

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
import yaml
from tqdm import tqdm

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )

        # print(test_set)
        # breakpoint()
        
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size= 1, #config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    print("config['test_dataset']", config['test_dataset'])
    for one_test_name in config['test_dataset']:
        print("one_test_name", one_test_name)
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

def test_gradcam(model, test_dataloader, device, model_name, exp_results_path, cam_method, s_layer, gotcha = False): #use_gradcam_plus_plus=False):
    model.eval()
    # NOTE: in testing UCF ONLY USES THE SHARED ENCODER AND HEAD!!!!!!!!!!!!
    # if 'ucf' in model_name and target_layer == 'spe':
    #     target_layers = [model.head_spe] # or model.head_sha]
    if 'ucf' in model_name and s_layer == 'head_sha':
        target_layers = [model.head_sha]
    elif 'ucf' in model_name and s_layer == 'encoder_c':
        target_layers =[model.encoder_c.conv4]
    elif s_layer == 'block_sha':
        target_layers = [model.block_sha]
    elif s_layer == 'encoder_f':
        target_layers = [model.encoder_f.conv4]
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        # slayer = 'model.'+target_layer
        # print("layer: ", slayer)
        # target_layers = [slayer]
        # print("target_layers: ", target_layers)

    print("target_layers: ", target_layers)        

    # # Initialize Grad-CAM or Grad-CAM++
    # if use_gradcam_plus_plus:
    #     cam = GradCAMPlusPlus(model=model, target_layers=[target_layer]) #, use_cuda=torch.cuda.is_available())
    # else:
    #     cam = GradCAM(model=model, target_layers=[target_layer]) #, use_cuda=torch.cuda.is_available())
    #     # target_layers is a list!!

    # Initialize the CAM-method
    if cam_method == 'gradcam':
        cam = GradCAM(model=model, target_layers=target_layers)
    elif cam_method == 'gradcam++':
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif cam_method == 'eigencam':
        cam = EigenCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unsupported CAM method: {cam_method}")

    
    keys = test_dataloader.keys()
    print("keys: ", keys) # dict_keys(['gotcha_occ_testing'])
    for key in keys:
        print(key)
        dataset_data_dict = test_dataloader[key].dataset.data_dict

        for i, data_dict in tqdm(enumerate(test_dataloader[key]), total=len(test_dataloader[key])):
            data, label = data_dict['image'], data_dict['label']
            label = torch.where(data_dict['label'] != 0, 1, 0)
            img_path = dataset_data_dict['image'][i]

            # data.requires_grad = True        
            
            print("img_path: ", img_path)

            if gotcha: 
                print("gotcha")
                if 'original' in img_path:
                    # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/original/hand_occlusion/00008_0.jpg
                    frame_id = img_path.split('/')[-1].split('.')[0]
                    challenge_id = img_path.split('/')[-2]
                    algo_id = img_path.split('/')[-3]
                    user_id = img_path.split('/')[-4]
                    print("original")
                    print("frame_id:", frame_id)
                    print("challenge_id: ", challenge_id)
                    print("algo_id: ", algo_id)
                    print("user_id: ", user_id)
                else:
                    # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/FSGAN/obj_occlusion/44/frame_0090301.jpg
                    frame_id = img_path.split('/')[-1].split('.')[0]
                    swap_id = img_path.split('/')[-2]
                    challenge_id = img_path.split('/')[-3]
                    algo_id = img_path.split('/')[-4]
                    user_id = img_path.split('/')[-5]

                    print("frame_id:", frame_id)
                    print("swap_id: ", swap_id)
                    print("challenge_id: ", challenge_id)
                    print("algo_id: ", algo_id)
                    print("user_id: ", user_id)
                    # user_id + algo_id + swap_id + challenge_id + frame_id

                # print("frame_id:", frame_id)
                # print("challenge_id: ", challenge_id)
                # print("algo_id: ", algo_id)
                # print("user_id: ", user_id)
            else: 
                # get info to save the image from image_paths
                frame_id = img_path.split('/')[-1].split('.')[0]
                challenge_id = img_path.split('/')[-2]
                algo_id = img_path.split('/')[-3]
                user_id = img_path.split('/')[-4]
                # user_id + algo_id + challenge_id + frame_id

            
            # breakpoint()

            # ----------------------------------------------------------------------------- #
            # Generate Grad-CAM for the single image (one image per batch)
            # None Targets
            # grayscale_cam = cam(input_tensor=data_dict, targets=None)[0] 
            # binartClassifierOutputTarget
            grayscale_cam = cam(input_tensor=data_dict, targets = [BinaryClassifierOutputTarget(data_dict['label'].item())])[0]
            # as discussed here: https://github.com/jacobgil/pytorch-grad-cam/issues/325 
            # BinaryClassifierOutputTarget -> if the net has only one output with a sigmoid
            # ----------------------------------------------------------------------------- #
            # Read the original image
            rgb_img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = np.float32(rgb_img) / 255
            rgb_img_resized = cv2.resize(rgb_img, (256, 256))

            # Overlay the heatmap on the original image
            cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
            # cam_image_path = f"{exp_results_path}/gradcam_{model_name}_{i}_{j}.png"
            cam_subfolders_path = f"{exp_results_path}/{user_id}/{algo_id}/"
            os.makedirs(cam_subfolders_path, exist_ok=True)
            if gotcha:
                if 'original' in img_path:
                    cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
                else: 
                    cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{swap_id}_{frame_id}.png"
            else: 
                cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
            cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            print(f"Grad-CAM image saved to: {cam_image_path}")

    print("done")


def get_args_parse():
    parser = argparse.ArgumentParser(description='Grad-CAM testing')
    parser.add_argument('--detector', type=str, default='ucf', help='Model name')
    parser.add_argument('--dataset', type=str, default='dfb_occ_testing', help='Dataset name')
    parser.add_argument('--tl', action='store_true', help='Use transfer learning model')
    parser.add_argument('--ft', action='store_true', help='Use fine-tuned model')
    # parser.add_argument('--tags', type=str, default='BinaryClassifierOutputTarget', help='Target type')
    parser.add_argument('--method', type=str, default='eigencam', choices=['gradcam', 'gradcam++', 'eigencam'], help='CAM method')
    parser.add_argument('--s-layer', type=str, 
                        default= "block_sha", #"encoder_c", 
                        help='UCF target layer')
    # parser.add_argument('--gradcam', action = 'store_false')

    return parser

def main():

    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    print(args)
    # breakpoint()
    gotcha = False
    # init_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # print("selected_layer: ", args.s_layer)
    # breakpoint() 

    # select the model (either MobileNet or EffNetB4_DFDC)
    if 'ucf' in args.detector.lower() :
        detector_yaml = './config/detector/ucf.yaml'
        if args.tl: 
            if 'gotcha_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_gotcha_occ_TL/ucf_2024-10-16-08-10-31/test/gotcha_occlusion/ckpt_best.pth' # ucf 32 imgs in trn/test & auc test metric 
                print(f"using TL model {args.detector}: {weights_path}")
                # gotcha = True

            # elif args.tl and args.test_dataset[0] == "gotcha_no_occlusion":
            elif 'gotcha_no_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-08-58-28/test/gotcha_no_occlusion/ckpt_best.pth'
                #'/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-07-29-26/test/gotcha_no_occlusion/ckpt_best.pth' # ucf trained with 32 imgs but auc as test metric
                print(f"using TL model {args.detector}: {weights_path}")
                # gotcha = True
            
            # config['pretrained'] = weights_path
            # print(config['pretrained'])
            # elif args.tl and args.test_dataset[0] == 'no_occlusion':
            elif 'dfb_no_occ' in args.detector:
                # if args.pretrained:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_dfb_no_occ_TL/ucf_2024-10-16-12-32-33/test/no_occlusion/ckpt_best.pth' # base UCF training with 100 ex per trn/tst and focal_loss 
                print(f"using TL model {args.detector}: {weights_path}")
                # gotcha = False

            elif 'dfb_occ' in args.detector:
                # if args.pretrained:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_dfb_occ_TL/ucf_2024-10-16-11-27-39/test/occlusion/ckpt_best.pth' # base UCF training (100 ex in trn/tst and focal_loss)
                print(f"using TL model {args.detector}: {weights_path}")
                # gotcha = False

            # elif args.tl and args.test_dataset[0] == "gotcha_occlusion":
            

        # elif args.ft and args.test_dataset[0] == 'occlusion':
        elif args.ft: 
            if 'gotcha_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/UCF_gotcha_occ_FT/ucf_2024-10-27-16-11-31/test/gotcha_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")
                # gotcha = True

            # elif args.ft and args.test_dataset[0] == "gotcha_no_occlusion":
            elif 'gotcha_no_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/UCF_gotcha_no_occ_FT/ucf_2024-10-27-17-45-13/test/gotcha_no_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")
                # gotcha = True

             # elif args.ft and args.test_dataset[0] == 'no_occlusion':
            elif 'dfb_no_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/UCF_dfb_no_occ_FT/ucf_2024-10-17-08-19-41/test/no_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")
                # gotcha = False

            elif 'dfb_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/UCF_dfb_occ_FT/ucf_2024-10-17-10-02-39/test/occlusion/ckpt_best.pth'
                print(f"using FT model {args.detector}: {weights_path}")
                # gotcha = False
        else:
           raise ValueError("Original model not available")
    else:
        raise ValueError(f"Unsupported model name: {args.detector}")


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

    # If arguments are provided, they will overwrite the yaml settings
    if args.dataset:
        config['test_dataset'] = [args.dataset]
    else:
        args.dataset = config['test_dataset']
    
    print("dataset:")
    print(args.dataset)
    print(config['test_dataset'])

    if 'gotcha' in args.dataset:
        gotcha = True
    else:
        gotcha = False

    print("gotcha: ", gotcha)
    # breakpoint()

    if weights_path:
        config['weights_path'] = weights_path


    # prepare the testing data loader
    print("prepare testing data")
    test_data_loaders = prepare_testing_data(config)
    print("done")


   

    # prepare the model (detector)
    print("loading the model...")

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

   
    print("model loaded!")

    if args.tl:
        model_name = args.detector + '_TL'
    else: 
        model_name = args.detector + '_FT'

    # exp_results_path = f'/home/rz/DeepfakeBench/training/results/gradcam_output/{model_name}_{args.dataset}_{args.method}_{args.s_layer}/'#_{args.tags}/'
    exp_results_path = f'/home/rz/DeepfakeBench/training/results/ucf_cross_models_gradcam/{model_name}_{args.dataset}_{args.method}_{args.s_layer}/'
    # exp_results_path = f'/home/rz/DeepfakeBench/training/results/test_gradcam/{model_name}_{args.dataset}_{args.method}_{args.s_layer}/'
    os.makedirs(exp_results_path, exist_ok=True)
    
    test_gradcam(model, test_data_loaders, device, model_name, exp_results_path, args.method, args.s_layer, gotcha) #, use_gradcam_plus_plus=False)
    
if __name__ == '__main__':
    main()