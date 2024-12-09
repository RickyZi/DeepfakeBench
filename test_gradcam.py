import torch
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Function
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        loss = F.nll_loss(output, target_class)
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image_path):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (299, 299))
    image = np.float32(image) / 255
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    return image

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) 
    cam = cam / np.max(cam)
    plt.imshow(cam)
    plt.show()

# ------------------------------------------ #
# for name, module in model.named_modules(): print(name, module)
# ------------------------------------------ #

# Load the Xception model
model = models.xception(pretrained=True)
target_layer = model.conv4  # Last convolutional layer before the fully connected layers
# ------------------------------------------ #
# ---------- GradCam target layer ---------- #
# ------------------------------------------ #
# MobileNetV2
# mobilenetv2_target_layer = model.features[-1]  # Last convolutional layer before the fully connected layers -> (18): Conv2dNormActivation
# model.classifier for the TL model
# ------------------------------------------ #
# EffNetB4
# efficientnetb4_target_layer = model.features[-1]  # Last convolutional layer before the fully connected layers
# classifier_target_layer = model.classifier 
# ------------------------------------------ #
# Xception
# target_layer = model.conv4  # Last convolutional layer before the fully connected layers
# classifier_target_layer = model.fc
# ------------------------------------------ #
# EffNetB4_FF and EffNetB4_DFDC
# efficientnetb4_target_layer = model.efficientnet._blocks[-1]  # Last convolutional layer before the fully connected layers
# model.classifier for the TL model
# ------------------------------------------ #
# Xception DFB
# target_layer = model.backbone.conv4  # Last convolutional layer before the fully connected layers (based on XceptionNet)
# classifier_target_layer = model.backbone.last_linear
# ------------------------------------------ #
# UCF DFB -> encoder_f or encoder_c blocks -> NEED TO CHECK THIS!!!!!
# encoder_f -> extracts the fingerprint = artifacts produced by different forgery techniques
# encoder_c -> extracts the content = info not directly related to the forgery (i.e. background, identity, and facial appearance)
# -> we might be more interested in finding which artifacts are learned by the net to detect the forgery -> encoder_f
# ucf_target_layer = model.encoder_f.conv4  # used both encoder to obtain fingerprint and content features
# classifier_target_layer = model.encoder_f.last_linear 
# ------------------------------------------ #

# Initialize Grad-CAM
grad_cam = GradCAM(model, target_layer)

# Preprocess the input image
image_path = 'path_to_your_image.jpg'
input_image = preprocess_image(image_path)

# Generate CAM
target_class = torch.tensor([0])  # Replace with the actual target class
cam = grad_cam.generate_cam(input_image, target_class)

# Show CAM on the image
original_image = cv2.imread(image_path, 1)
original_image = cv2.resize(original_image, (299, 299))
show_cam_on_image(original_image, cam)