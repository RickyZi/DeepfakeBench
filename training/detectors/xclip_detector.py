'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XCLIPDetector

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
@inproceedings{ma2022x,
  title={X-clip: End-to-end multi-grained contrastive learning for video-text retrieval},
  author={Ma, Yiwei and Xu, Guohai and Sun, Xiaoshuai and Yan, Ming and Zhang, Ji and Ji, Rongrong},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={638--647},
  year={2022}
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

import loralib as lora

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='xclip')
class XCLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.fc_norm = nn.LayerNorm(768)
        # self.temporal_module = self.build_temporal_module(config)
        self.head = nn.Linear(768, 2)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config):
        from transformers import XCLIPVisionModel
        backbone = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        return backbone
    
    def build_temporal_module(self, config):
        model = nn.LSTM(input_size=2048, hidden_size=512, num_layers=3, batch_first=True)
        return model
    
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        # b, t, c, h, w = data_dict['image'].shape
        # frame_input = data_dict['image'].reshape(-1, c, h, w)
        # # get frame-level features
        # frame_level_features = self.backbone.features(frame_input)
        # frame_level_features = F.adaptive_avg_pool2d(frame_level_features, (1, 1)).reshape(b, t, -1)
        # # get video-level features
        # video_level_features = self.temporal_module(frame_level_features)[0][:, -1, :]

        batch_size, num_frames, num_channels, height, width = data_dict['image'].shape
        pixel_values = data_dict['image'].reshape(-1, num_channels, height, width)
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        sequence_output = outputs['pooler_output'].reshape(batch_size, num_frames, -1)
        video_level_features = self.fc_norm(sequence_output.mean(1))
        return video_level_features

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        metric_batch_dict = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0}
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
        # if inference:
        #     self.prob.extend(
        #         pred_dict['prob']
        #         .detach()
        #         .squeeze()
        #         .cpu()
        #         .numpy()
        #     )
        #     self.label.extend(
        #         data_dict['label']
        #         .detach()
        #         .squeeze()
        #         .cpu()
        #         .numpy()
        #     )
        #     # deal with acc
        #     _, prediction_class = torch.max(pred, 1)
        #     correct = (prediction_class == data_dict['label']).sum().item()
        #     self.correct += correct
        #     self.total += data_dict['label'].size(0)

        return pred_dict
