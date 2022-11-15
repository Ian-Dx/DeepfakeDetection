# from dxtorchutils.ImageClassification.validate import ValidateVessel
from torch.utils.data import DataLoader
from fan import FAN
from dataset import MyDataset
from dxtorchutils.utils.metrics import precision_macro, recall_macro
from tesing import ValidateVessel
from dxtorchutils.ImageClassification.models import ResNet50


def validate_():
    dataset = MyDataset(process_type="testing", reshape_size=(224,224),dataset_type="c40")
    dataloader = DataLoader(dataset, 1200, True)

    vv = ValidateVessel(dataloader, ResNet50())

    vv.gpu()

    vv.load_model_para("models/resnet50.pth")
    # vv.set_tensorboard_dir("./logger/resnet50_val_40")


    vv.validate()


validate_()
