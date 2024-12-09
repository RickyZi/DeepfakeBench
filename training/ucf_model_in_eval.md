model in eval
UCFDetector(
  (encoder_f): Xception(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (block1): Block(
      (skip): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): SeparableConv2d(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (pointwise): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): SeparableConv2d(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block2): Block(
      (skip): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block3): Block(
      (skip): Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block4): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block5): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block6): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block7): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block8): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block9): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block10): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block11): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block12): Block(
      (skip): Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (conv3): SeparableConv2d(
      (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
      (pointwise): Conv2d(1024, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (bn3): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv4): SeparableConv2d(
      (conv1): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
      (pointwise): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (bn4): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (last_linear): Linear(in_features=2048, out_features=2, bias=True)
    (adjust_channel): Sequential(
      (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (encoder_c): Xception(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (block1): Block(
      (skip): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): SeparableConv2d(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (pointwise): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): SeparableConv2d(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block2): Block(
      (skip): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
          (pointwise): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block3): Block(
      (skip): Conv2d(256, 728, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (pointwise): Conv2d(256, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (block4): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block5): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block6): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block7): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block8): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block9): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block10): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block11): Block(
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        (7): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (8): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (block12): Block(
      (skip): Conv2d(728, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (skipbn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (rep): Sequential(
        (0): ReLU()
        (1): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 728, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (2): BatchNorm2d(728, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU(inplace=True)
        (4): SeparableConv2d(
          (conv1): Conv2d(728, 728, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=728, bias=False)
          (pointwise): Conv2d(728, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        (5): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
    )
    (conv3): SeparableConv2d(
      (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
      (pointwise): Conv2d(1024, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (bn3): BatchNorm2d(1536, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (conv4): SeparableConv2d(
      (conv1): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536, bias=False)
      (pointwise): Conv2d(1536, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (bn4): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (last_linear): Linear(in_features=2048, out_features=2, bias=True)
    (adjust_channel): Sequential(
      (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (lr): LeakyReLU(negative_slope=0.01, inplace=True)
  (do): Dropout(p=0.2, inplace=False)
  (pool): AdaptiveAvgPool2d(output_size=1)
  (con_gan): Conditional_UNet(
    (upsample): Upsample(scale_factor=2.0, mode='bilinear')
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (dropout): Dropout(p=0.3, inplace=False)
    (adain3): AdaIN()
    (adain2): AdaIN()
    (adain1): AdaIN()
    (dconv_up3): Sequential(
      (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
    )
    (dconv_up2): Sequential(
      (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
    )
    (dconv_up1): Sequential(
      (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
    )
    (conv_last): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1))
    (up_last): Upsample(scale_factor=4.0, mode='bilinear')
    (activation): Tanh()
  )
  (head_spe): Head(
    (do): Dropout(p=0.2, inplace=False)
    (pool): AdaptiveAvgPool2d(output_size=1)
    (mlp): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Linear(in_features=512, out_features=2, bias=True)
    )
  )
  (head_sha): Head(
    (do): Dropout(p=0.2, inplace=False)
    (pool): AdaptiveAvgPool2d(output_size=1)
    (mlp): Sequential(
      (0): Linear(in_features=256, out_features=512, bias=True)
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Linear(in_features=512, out_features=2, bias=True)
    )
  )
  (block_spe): Conv2d1x1(
    (conv2d): Sequential(
      (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (3): LeakyReLU(negative_slope=0.01, inplace=True)
      (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (block_sha): Conv2d1x1( 
    (conv2d): Sequential(
      (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): LeakyReLU(negative_slope=0.01, inplace=True)
      (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (3): LeakyReLU(negative_slope=0.01, inplace=True)
      (4): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)