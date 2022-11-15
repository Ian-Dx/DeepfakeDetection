# from dxtorchutils.ImageClassification.validate import ValidateVessel
from torch.utils.data import DataLoader
from fan import FAN
from dataset import MyDataset
from dxtorchutils.utils.metrics import precision_macro, recall_macro
from tesing import ValidateVessel
from dxtorchutils.ImageClassification.models import ResNet50


def validate_():
    dataset = MyDataset(process_type="testing",dataset_type="c23")
    dataloader = DataLoader(dataset, 1200, True)

    vv = ValidateVessel(dataloader, FAN(2))

    vv.gpu()

    vv.load_model_para("models/fan4.pth")
    # vv.set_tensorboard_dir("./logger/fan_val_23")

    vv.validate()



validate_()
