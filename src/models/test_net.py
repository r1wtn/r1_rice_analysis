import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT, image_size=224)
        vit.heads[0] = nn.Linear(768, 224)
        self.model = vit



    def forward(self, x):
        x = self.model(x)
        return x

class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.fc = nn.Linear(224, 224)
     
    def forward(self, x):
        x = self.fc(x)
        return x