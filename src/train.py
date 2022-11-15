import time

from fan import FAN
from torch.utils.data import DataLoader
from dataset import MyDataset
from dxtorchutils.utils.train import TrainVessel
from dxtorchutils.ImageClassification.models import ResNet50
from noise_net import NoiseNet
import torch


def train(model, is_continue=False):
    dataset = MyDataset(reshape_size=(224, 224))
    dataloader = DataLoader(dataset, 200, True)

    tv = TrainVessel(dataloader, model)
    # tv.opt = torch.optim.SGD(model.parameters(), 0.0001)
    tv.gpu()
    if is_continue:
        tv.load_model_para("models/{}_resume.pth".format(model.__class__.__name__.lower()))
        tv.save_model_to("models/{}_resume2.pth".format(model.__class__.__name__.lower()))
        tv.set_tensorboard_dir("./logger/{}_resume3".format(model.__class__.__name__.lower()))

    else:
        tv.save_model_to("models/{}_lite_tanh.pth".format(model.__class__.__name__.lower()))
        tv.set_tensorboard_dir("./logger/{}_lite_tanh".format(model.__class__.__name__.lower()))
    tv.eval_num = 1000

    tv.train()


dataset = MyDataset(reshape_size=(224, 224))
dataloader = DataLoader(dataset, 200, True)

tv = TrainVessel(dataloader, NoiseNet())

tv.gpu()

tv.save_model_to("models/filter2.pth")
tv.set_tensorboard_dir("./logger/filter2")
tv.eval_num = 1000


tv.train()




