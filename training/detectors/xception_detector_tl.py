'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):
    def __init__(self, config, tl = False):
        super().__init__()
        self.config = config
        self.tl = tl
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0

        # breakpoint()

        # ----------------------------------- #
        if self.tl:
            print("Freezing the backbone weights")
            self.freeze_backbone()
        # ----------------------------------- #
        
    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k} # remove classifier from state_dict 
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        # print(backbone)
        
        return backbone
    
    # ----------------------------------- #
    def freeze_backbone(self):
        # self.check_model_gradient()
        # breakpoint()

        for param in self.backbone.parameters():
            param.requires_grad = False
        
        print(self.backbone.last_linear)

        for param in self.backbone.last_linear.parameters():
            param.requires_grad = True
        
        # print("classifier: " self.classifier)

        self.check_model_gradient()
        # breakpoint()
        
        # # Ensure the classifier layer is trainable
        # if hasattr(self.backbone, 'classifier'):
            
        #     print("classifier")
        #     print(self.backbone.classifier)

        #     # for param in self.backbone.classifier.parameters():
        #     #     param.requires_grad = True
        # elif hasattr(self.backbone, 'fc'):
        #     print("fc")
        #     print(self.backbone.fc)
        #     # for param in self.backbone.fc.parameters():
        #     #     param.requires_grad = True
        # elif hasattr(self.backbone, 'last_linear'):
        #     print("last_linear")
        #     print(self.backbone.last_linear)
        # else:
        #     raise AttributeError("The backbone model does not have a 'classifier' or 'fc' attribute.")


    def check_model_gradient(self):
        # Check if the gradient is active for the FC layer
        for name, param in self.backbone.named_parameters():
            print(f"Layer: {name}, requires_grad: {param.requires_grad}")
    # ----------------------------------- #
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        overall_loss = loss
        loss_dict = {'overall': overall_loss, 'cls': loss,}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
