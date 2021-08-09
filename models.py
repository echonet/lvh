import torch
import torchvision
from pathlib import Path


class PlaxModel(torch.nn.Module):

    """Model used for prediction of PLAX measurement points.
    Output channels correspond to heatmaps for the endpoints of
    measurements of interest.
    """

    def __init__(self, 
            measurements=['LVPW', 'LVID', 'IVS'], 
        ) -> None:
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(measurements) + 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])


class ClassificationModel(torch.nn.Module):

    """Binary video classification used to classify heart conditions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.video.r3d_18()
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    
    def forward(self, x):
        return self.model(x)
