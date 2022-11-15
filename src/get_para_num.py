import torch
from fan import FAN
from dxtorchutils.ImageClassification.models import ResNet50
from torchsummary import summary
from torchvision.models import resnet50
from noise_net import NoiseNet

def get_para(model):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = model.to(device)
    model.cuda()
    summary(model, input_size=(3, 224, 224))


get_para(NoiseNet())
# get_para(FAN(2))

# print(ResNet50())
# print(resnet50())


