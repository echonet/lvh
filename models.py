import torch
import torchvision
from pathlib import Path


class PlaxModel(torch.nn.Module):

    def __init__(self, 
            measurements=['LVPW', 'LVID', 'IVS'], 
        ) -> None:
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(measurements) + 1)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])


class ClassificationModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = torchvision.models.video.r3d_18()
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # weights_path = Path.cwd() / 'wandb' / 'run-20210408_112436-6ddw6vl7' / 'files' / 'trained_model.pt'
    weights_path = Path.cwd() / 'classification' / 'wandb' / 'run-20210616_123249-2y9ls3j9' / 'files' / 'best_auc_model.pt'
    model = ClassificationModel()
    model.to('cpu')
    print(model.load_state_dict(torch.load(weights_path, map_location='cpu')))