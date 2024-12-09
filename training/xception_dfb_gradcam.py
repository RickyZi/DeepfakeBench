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

def test_gradcam(model, test_dataloader, device, model_name, exp_results_path, cam_method, gotcha = False): #use_gradcam_plus_plus=False):
    model.eval()
    # correct = 0
    # total = 0

    # Define the target layer for Grad-CAM
    # if 'mnetv2' in model_name:
    #     if num_layers == 1:
    #         target_layers = model.features[-1]
    #     elif num_layers == 2:
    #         target_layers = [model.features[-1], model.features[-2]]
    #     elif num_layers == 3:
    #         target_layers = [model.features[-1], model.features[-2], model.features[-3]]
    #     else:
    #         raise ValueError(f"Unsupported number of target layers: {num_layers}")
    #     # target_layer = model.features[-1]
    # elif 'effnetb4_dfdc' in model_name:
    #     # target_layers = [model.efficientnet._blocks[-1]] 
    #     if num_layers == 1:
    #         target_layers = [model.efficientnet._blocks[-1]] 
    #     elif num_layers == 2:
    #         target_layers = [model.efficientnet._blocks[-1], model.efficientnet._blocks[-2]]
    #     elif num_layers == 3:
    #         target_layers = [model.efficientnet._blocks[-1], model.efficientnet._blocks[-2], model.efficientnet._blocks[-3]]
    #     else:
    #         raise ValueError(f"Unsupported number of target layers: {num_layers}")

        #model.features[-1]
    if 'xception' in model_name:
        target_layers = [model.backbone.conv4]
    else:
        raise ValueError(f"Unsupported target_layers name: {model_name}")
    
    # NOTE: in testing UCF ONLY USES THE SHARED ENCODER AND HEAD!!!!!!!!!!!!
    # if 'ucf' in model_name and target_layer == 'spe':
    #     target_layers = [model.head_spe] # or model.head_sha]
    # if 'ucf' in model_name and target_layer == 'sha':
    #     target_layers = [model.head_sha]
    # elif 'ucf' in model_name and target_layer == 'encoder_c':
    #     target_layers =[model.encoder_c.conv4]
    # elif target_layer == 'block_sha':
    #     target_layers = [model.block_sha]
    # else:
    #     raise ValueError(f"Unsupported model name: {model_name}")
    #     # slayer = 'model.'+target_layer
    #     # print("layer: ", slayer)
    #     # target_layers = [slayer]
    #     # print("target_layers: ", target_layers)
        

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

    # for i, (images, labels, image_paths) in enumerate(test_dataloader):
    #     print(f"Batch {i}") # print the batch number -> 1 image per batch (batch_size = 1)
    #     print(images.size()) # (batch_size, 3, 224, 224)
    #     print(labels.size()) # (batch_size, 1)

    #     # stop after 10 images processed
    #     if i == 10:
    #         break

    #     # if i == 0:
    #     #     print("image_paths", image_paths)
    #         # /media/data/rz_dataset/users_face_occlusion/testing/user_300229/facedancer_faces/hand_occlusion_1/frame210.jpg
        

    #     for j in range(images.size(0)):
    #         image = images[j].unsqueeze(0).to(device)
    #         print("image.shape", image.shape)
    #         label = labels[j].unsqueeze(0).to(device)
    #         print("label.shape", label.shape)
    #         print("label.item()", label.item())
            
    #         if gotcha: 
    #             if 'original' in image_paths[j]:
    #                 # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/original/hand_occlusion/00008_0.jpg
    #                 frame_id = image_paths[j].split('/')[-1].split('.')[0]
    #                 challenge_id = image_paths[j].split('/')[-2]
    #                 algo_id = image_paths[j].split('/')[-3]
    #                 user_id = image_paths[j].split('/')[-4]
    #             else:
    #                 # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/FSGAN/obj_occlusion/44/frame_0090301.jpg
    #                 frame_id = image_paths[j].split('/')[-1].split('.')[0]
    #                 swap_id = image_paths[j].split('/')[-2]
    #                 challenge_id = image_paths[j].split('/')[-3]
    #                 algo_id = image_paths[j].split('/')[-4]
    #                 user_id = image_paths[j].split('/')[-5]
    #         else: 
    #             # get info to save the image from image_paths
    #             frame_id = image_paths[j].split('/')[-1].split('.')[0]
    #             challenge_id = image_paths[j].split('/')[-2]
    #             algo_id = image_paths[j].split('/')[-3]
    #             user_id = image_paths[j].split('/')[-4]

    #         # print(image_paths)
    #         output = model(image)
    #         _, predicted = torch.max(output.data, 1)
    #         total += label.size(0)
    #         correct += (predicted == label).sum().item()

    #         # Generate Grad-CAM for the single image (one image per batch)
    #         # grayscale_cam = cam(input_tensor=image, targets=None)[0] #[ClassifierOutputTarget(label.item())])[0]
    #         grayscale_cam = cam(input_tensor=image, targets = [BinaryClassifierOutputTarget(label.item())])[0]
    #         # as discussed here: https://github.com/jacobgil/pytorch-grad-cam/issues/325 
    #         # BinaryClassifierOutputTarget -> if the net has only one output with a sigmoid

    #         # Normalize the heatmap to the range [0, 255] and convert to uint8
    #         # grayscale_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))
    #         # print("resized grayscale img", type(grayscale_cam))
    #         # grayscale_cam = np.uint8(255 * grayscale_cam)
    #         # print("np.uint8 grayscaleimg", type(grayscale_cam))
    #         # grayscale_cam = np.uint8(255 * grayscale_cam)

    #         # Resize the heatmap to match the original image size
    #         # grayscale_cam = cv2.resize(grayscale_cam, (images[j].shape[2], images[j].shape[1]))

    #         # Read the original image
    #         rgb_img = cv2.imread(image_paths[j])
    #         rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    #         rgb_img = np.float32(rgb_img) / 255
    #         rgb_img_resized = cv2.resize(rgb_img, (224, 224))

    #         # Overlay the heatmap on the original image
    #         cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
    #         # cam_image_path = f"{exp_results_path}/gradcam_{model_name}_{i}_{j}.png"
    #         cam_subfolders_path = f"{exp_results_path}/{user_id}/{algo_id}/"
    #         os.makedirs(cam_subfolders_path, exist_ok=True)
    #         if gotcha:
    #             if 'original' in image_paths[j]:
    #                 cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
    #             else: 
    #                 cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{swap_id}_{frame_id}.png"
    #         else: 
    #             cam_image_path = f"{cam_subfolders_path}/{challenge_id}_{frame_id}.png"
    #         cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    #         print(f"Grad-CAM image saved to: {cam_image_path}")


    keys = test_dataloader.keys()
    print("keys: ", keys) # dict_keys(['gotcha_occ_testing'])
    for key in keys:
        print(key)
        dataset_data_dict = test_dataloader[key].dataset.data_dict

        for i, data_dict in tqdm(enumerate(test_dataloader[key]), total=len(test_dataloader[key])):
            data, label = data_dict['image'], data_dict['label']
            label = torch.where(data_dict['label'] != 0, 1, 0)
            img_path = dataset_data_dict['image'][i]
            print('img_path', img_path)
            print("data: ", data) # img - tensor
            # print("len(data): ", len(data))
            # print("type(data): ", type(data)) #tensor
            print("label", label) # label - tensor
            print("label.item()", label.item())
            print("type(data): ", type(data))
            # print("len(label):", len(label)) # tensor
            # print("label.item(): ", label.item()) # int (0, 1)
            # label = torch.where(data_dict['label'] != 0, 1, 0)
            # print("label_torch: ", label)
            # move data to GPU
            data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
            # img_tensor = data_dict['image']
            # label_tensor = label.to(device)
            print("data_dict: ", data_dict)
            print("data_dict['image']", data_dict['image'])
            print("data_dict['label']", data_dict['label'])
            # print("data_dict['image'].shape", data_dict['image'].shape)
            # print("data_dict['label'].shape", data_dict['label'].shape)
            # print("data_dict['image']", data_dict['image'])
            # if i == 0: break
            print("before gradcam")
            # breakpoint()

            if gotcha: 
                if 'original' in img_path:
                    # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/original/hand_occlusion/00008_0.jpg
                    frame_id = img_path.split('/')[-1].split('.')[0]
                    challenge_id = img_path.split('/')[-2]
                    algo_id = img_path.split('/')[-3]
                    user_id = img_path.split('/')[-4]
                else:
                    # image_path /home/rz/rz-test/bceWLL_test/rand_imgs_test/rand_imgs_gotcha/occ_testing/33/FSGAN/obj_occlusion/44/frame_0090301.jpg
                    frame_id = img_path.split('/')[-1].split('.')[0]
                    swap_id = img_path.split('/')[-2]
                    challenge_id = img_path.split('/')[-3]
                    algo_id = img_path.split('/')[-4]
                    user_id = img_path.split('/')[-5]
            else: 
                # get info to save the image from image_paths
                frame_id = img_path.split('/')[-1].split('.')[0]
                challenge_id = img_path.split('/')[-2]
                algo_id = img_path.split('/')[-3]
                user_id = img_path.split('/')[-4]


            # ----------------------------------------------------------------------------- #
            # Generate Grad-CAM for the single image (one image per batch)
            # None Targets
            # grayscale_cam = cam(input_tensor=data_dict['image'], targets=None)[0] 
            
            # binartClassifierOutputTarget
            grayscale_cam = cam(input_tensor=data_dict['image'], targets = [BinaryClassifierOutputTarget(data_dict['label'].item())])[0]
            print("after gradcam")
            # as discussed here: https://github.com/jacobgil/pytorch-grad-cam/issues/325 
            # BinaryClassifierOutputTarget -> if the net has only one output with a sigmoid
            # ----------------------------------------------------------------------------- #
            # grayscale_cam = cam(input_tensor=data, targets = [BinaryClassifierOutputTarget(label.item())])[0]
            # Normalize the heatmap to the range [0, 255] and convert to uint8
            # grayscale_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))
            # print("resized grayscale img", type(grayscale_cam))
            # grayscale_cam = np.uint8(255 * grayscale_cam)
            # print("np.uint8 grayscaleimg", type(grayscale_cam))
            # grayscale_cam = np.uint8(255 * grayscale_cam)

            # Resize the heatmap to match the original image size
            # grayscale_cam = cv2.resize(grayscale_cam, (images[j].shape[2], images[j].shape[1]))

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

            # print("data_dict['image']: ", data_dict['image'])
            # print("data_dict['label']: ", data_dict['label'])

        # for i in range(len(data_dict['image'])):
        #     image = data_dict['image'][i] # path_to_img (str)
        #     label = data_dict['label'][i] # label (int)
        #     print("image", image)
        #     print("label", label)
        #     print("type(image)", type(image))
        #     print("type(label)", type(label))

            # if i == 0: break

        

    # accuracy = 100 * correct / total
    # print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    # return accuracy


def get_args_parse():
    parser = argparse.ArgumentParser(description='Grad-CAM testing')
    parser.add_argument('--detector', type=str, default='xception', help='Model name')
    parser.add_argument('--dataset', type=str, default='dfb_occ_testing', help='Dataset name')
    parser.add_argument('--tl', action='store_true', help='Use transfer learning model')
    parser.add_argument('--ft', action='store_true', help='Use fine-tuned model')
    # parser.add_argument('--tags', type=str, default='BinaryClassifierOutputTarget', help='Target type')
    parser.add_argument('--method', type=str, default='gradcam++', choices=['gradcam', 'gradcam++', 'eigencam'], help='CAM method')
    # parser.add_argument('--target-layer', type=str, default="conv4", help='UCF target layer')
    # parser.add_argument('--gradcam', action = 'store_false')

    return parser

def main():

    parser = get_args_parse() # get the arguments from the command line 
    args, unknown = parser.parse_known_args() # parse the known arguments and ignore the unknown ones
    gotcha = False
    # init_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    # # select the model (either MobileNet or EffNetB4_DFDC)
    # if args.model == 'ucf':
    #     detector_yaml = './config/detector/ucf.yaml'
    #     # model_name = 'ucf'

    #     if args.tl: 
    #         if args.dataset == 'gotcha_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_gotcha_occ_TL/ucf_2024-10-16-08-10-31/test/gotcha_occlusion/ckpt_best.pth' # ucf 32 imgs in trn/test & auc test metric 
    #             print(f"using TL model {args.model}: {weights_path}")
    #             gotcha = True

    #         # elif args.tl and args.test_dataset[0] == "gotcha_no_occlusion":
    #         elif args.dataset == 'gotcha_no_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-08-58-28/test/gotcha_no_occlusion/ckpt_best.pth'
    #             #'/home/rz/DeepfakeBench/training/results/TL/UCF_gotcha_no_occ_TL/ucf_2024-10-16-07-29-26/test/gotcha_no_occlusion/ckpt_best.pth' # ucf trained with 32 imgs but auc as test metric
    #             print(f"using TL model {args.model}: {weights_path}")
    #             gotcha = True
    #         elif args.dataset == 'dfb_occ_testing':
    #             # if args.pretrained:
    #             weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_dfb_occ_TL/ucf_2024-10-16-11-27-39/test/occlusion/ckpt_best.pth' # base UCF training (100 ex in trn/tst and focal_loss)
    #             print(f"using TL model {args.model}: {weights_path}")
    #         # config['pretrained'] = weights_path
    #         # print(config['pretrained'])
    #         # elif args.tl and args.test_dataset[0] == 'no_occlusion':
    #         elif args.dataset == 'dfb_no_occ_testing':
    #             # if args.pretrained:
    #             weights_path = '/media/data/model_exp_results/DFB/TL/TL/UCF_dfb_no_occ_TL/ucf_2024-10-16-12-32-33/test/no_occlusion/ckpt_best.pth' # base UCF training with 100 ex per trn/tst and focal_loss 
    #             print(f"using TL model {args.model}: {weights_path}")

    #         # elif args.tl and args.test_dataset[0] == "gotcha_occlusion":
            

    #     # elif args.ft and args.test_dataset[0] == 'occlusion':
    #     elif args.ft: 
    #         if args.dataset  == 'gotcha_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/FT/UCF_gotcha_occ_FT/ucf_2024-10-27-16-11-31/test/gotcha_occlusion/ckpt_best.pth' 
    #             print(f"using FT model {args.model}: {weights_path}")
    #             gotcha = True

    #         # elif args.ft and args.test_dataset[0] == "gotcha_no_occlusion":
    #         elif args.dataset == 'gotcha_no_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/FT/UCF_gotcha_no_occ_FT/ucf_2024-10-27-17-45-13/test/gotcha_no_occlusion/ckpt_best.pth' 
    #             print(f"using FT model {args.model}: {weights_path}")
    #             gotcha = True
    #         elif args.dataset == 'dfb_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/FT/UCF_dfb_occ_FT/ucf_2024-10-17-10-02-39/test/occlusion/ckpt_best.pth'
    #             print(f"using FT model {args.model}: {weights_path}")

    #         # elif args.ft and args.test_dataset[0] == 'no_occlusion':
    #         elif args.dataset == 'dfb_no_occ_testing':
    #             weights_path = '/media/data/model_exp_results/DFB/FT/UCF_dfb_no_occ_FT/ucf_2024-10-17-08-19-41/test/no_occlusion/ckpt_best.pth' 
    #             print(f"using FT model {args.model}: {weights_path}")

    #     else:
    #        raise ValueError("Original model not available")
    # else:
    #     raise ValueError(f"Unsupported model name: {args.model}")

    if 'xception' in args.detector:
        detector_yaml = './config/detector/xception.yaml'       
        model_name = 'xception'
        # if args.tl and args.test_dataset[0] == "occlusion":
        # elif args.tl and args.test_dataset[0] == "gotcha_occlusion":
        if args.tl : 
            if 'gotcha_occ' in args.detector:
                # model trained with 100 imgs per class
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/Xception_gotcha_occ_TL_def_frame_num/xception_2024-10-15-12-40-10/test/gotcha_occlusion/ckpt_best.pth' # test 32 frames for trn/tst (default frame_num)'
                print(f"using TL model {args.detector}: {weights_path}")
                gotcha = True

            # elif args.tl and args.test_dataset[0] == "gotcha_no_occlusion":
            elif args.tl and 'gotcha_no_occ' in args.detector:
                # model trained with 200 imgs per class
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/Xception_gotcha_no_occ_TL/xception_2024-10-15-13-57-27/test/gotcha_no_occlusion/ckpt_best.pth' # test 32 frames for trn/tst
                #'/home/rz/DeepfakeBench/training/results/TL/Xception_dfb_gotcha_no_occ_TL/xception_2024-10-03-13-13-29/test/gotcha_no_occlusion/ckpt_best.pth'
                print(f"using TL model {args.detector}: {weights_path}")
                gotcha = True
            elif args.tl and 'occ' in args.detector: 
                # if args.pretrained:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/Xception_dfb_occ_TL/xception_2024-09-20-13-28-49/test/occlusion/ckpt_best.pth' # focal_loss [USE THIS!!!!]
                print(f"using TL model {args.detector}: {weights_path}")
                # config['pretrained'] = weights_path
                # print(config['pretrained'])
            # elif args.tl and args.test_dataset[0] == "no_occlusion":
            elif args.tl and 'no_occ' in args.detector:
                # if args.pretrained:
                weights_path = '/media/data/model_exp_results/DFB/TL/TL/Xception_dfb_no_occ_TL/xception_2024-10-15-09-55-04/test/no_occlusion/ckpt_best.pth' # focal_loss [USE THIS!!!!]
                print(f"using TL model {args.detector}: {weights_path}")
                # config['pretrained'] = weights_path
                # print(config['pretrained'])
        elif args.ft: 
            # elif args.ft and args.test_dataset[0] == "gotcha_occlusion":
            if 'gotcha_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/Xception_gotcha_occ_FT/xception_2024-10-22-09-51-18/test/gotcha_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")
                gotcha = True

            # elif args.ft and args.test_dataset[0] == "gotcha_no_occlusion":
            elif 'gotcha_no_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/Xception_gotcha_no_occ_FT/xception_2024-10-22-10-29-24/test/gotcha_no_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")
                gotcha = True
            # elif args.ft and args.test_dataset[0] == 'occlusion':
            elif  'occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/Xception_dfb_occ_FT/xception_2024-09-11-11-41-36/test/occlusion/ckpt_best.pth'
                #'/home/rz/DeepfakeBench/training/results/FT/Xception_dfb_occ_FT/xception_2024-09-17-12-32-52/test/occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")

            # elif args.ft and args.test_dataset[0] == 'no_occlusion':
            elif  'no_occ' in args.detector:
                weights_path = '/media/data/model_exp_results/DFB/FT/Xception_dfb_no_occ_FT/xception_2024-09-11-12-31-09/test/no_occlusion/ckpt_best.pth' 
                print(f"using FT model {args.detector}: {weights_path}")

        else:
           raise ValueError("Original model not available")
        
    else:
        raise ValueError(f"Unsupported model name: {args.model}")


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

    if weights_path:
        config['weights_path'] = weights_path


    # prepare the testing data loader
    print("prepare testing data")
    test_data_loaders = prepare_testing_data(config)
    print("done")


    # keys = test_data_loaders.keys()

    # for key in keys:
    #     print(key)
    #     data_dict = test_data_loaders[key].dataset.data_dict
    #     # print("data_dict", data_dict) 
    #     # data_dict.keys() -> [image, label]
    #     # image -> contains image path
    #     # label contains the label (int)
    # # breakpoint()
    # print("data_dict['image'][0]", data_dict['image'][0]) # path to img
    # print("data_dict['label'][0]", data_dict['label'][0]) # label
    # breakpoint()
    
    # # print("len()")

    #     for i, data_dict in tqdm(enumerate(test_data_loaders), total=len(test_data_loaders)):
    #         data = data_dict['image']
    #         label= data_dict['label']
    #     #     image_paths = data_dict['img_paths']

    #         print("img: ", data)
    #         print("label: ", label)
    # #     print("img_path", image_paths)

    #     # label = torch.where(data_dict['label'] != 0, 1, 0)
    #     # move data to GPU
    #     # data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    # breakpoint()


    # prepare the model (detector)
    print("loading the model...")

    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)

    # print(model)
    # breakpoint()

    # add weights_path to log file
    # logger.info(f"model weights path: {weights_path}")

    # model = ModifiedModel(model, 2).to(device)
    print("model loaded!")

    if args.tl:
        model_name = args.detector + '_TL'
    else: 
        model_name = args.detector + '_FT'

    exp_results_path = f'/home/rz/DeepfakeBench/training/results/gradcam_output/test_{model_name}_{args.dataset}_{args.method}/'#_{args.tags}/'
    
    os.makedirs(exp_results_path, exist_ok=True)
    
    test_gradcam(model, test_data_loaders, device, model_name, exp_results_path, args.method, gotcha) #, use_gradcam_plus_plus=False)
    # model, test_dataloader, device, model_name, exp_results_path, cam_method, target_layer, gotcha = False

    # transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # if milan_aug: 
    #     # add the Milan Augmentation
    #     milan_transforms = milan_test_transf()
    # else: 
    #     test_transform = transforms.Compose([
    #         transforms.Resize((256,256)),
    #         transforms.CenterCrop((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    # # select the test dataset
    # if args.dataset == 'thesis_occ':
    #     test_dataset = FaceImagesDataset('/media/data/rz_dataset/users_face_occlusion/testing/', test_transform)
    # elif args.dataset == 'thesis_no_occ':
    #     test_dataset = FaceImagesDataset('/media/data/rz_dataset/users_face_no_occlusion/testing/', test_transform)
    # elif args.dataset == 'gotcha_occ':
    #     test_dataset = FaceImagesDataset('/media/data/rz_dataset/gotcha/balanced_gotcha/occlusion/testing/', test_transform)
    #     gotcha = True
    # elif args.dataset == 'gotcha_no_occ':
    #     test_dataset = FaceImagesDataset('/media/data/rz_dataset/gotcha/balanced_gotcha/no_occlusion/testing', test_transform)
    #     gotcha = True
    # else: 
    #     raise ValueError(f"Unsupported dataset name: {args.dataset}")
    # test_dataset = FaceImagesDataset(directory='/content/drive/MyDrive/WORK/test_gradcam/rand_imgs/thesis_occ/', transform=transform)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # exp_results_path = f'/content/drive/MyDrive/WORK/test_gradcam/gradcam_output/test_{model_name}_gradcam_BinaryClassifierOutputTarget_gradcam/'
    
    # exp_results_path = f'/content/drive/MyDrive/WORK/test_gradcam/gradcam_output/test_{args.model}_{'TL' if args.tl else 'FT'}_{args.dataset}_{args.method}_target_layers_{args.num_layers}/' #_{args.tags}/'
    # os.makedirs(exp_results_path, exist_ok=True)
    
    # test_gradcam(model, test_dataloader, device, args.model, exp_results_path, args.method, gotcha) #, use_gradcam_plus_plus=False)

if __name__ == '__main__':
    main()