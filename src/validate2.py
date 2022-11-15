# from dxtorchutils.ImageClassification.validate import ValidateVessel
from torch.utils.data import DataLoader
from fan import FAN
from dataset import MyDataset
from dxtorchutils.utils.metrics import precision_macro, recall_macro
from tesing import ValidateVessel
from dxtorchutils.ImageClassification.models import ResNet50


def validate_():
    dataset = MyDataset(process_type="testing")
    dataloader = DataLoader(dataset, 1200, True)

    vv = ValidateVessel(dataloader, FAN(2))

    vv.gpu()

    vv.load_model_para("fan.pth")
    vv.set_tensorboard_dir("./logger/fan_split_val")
    vv.add_metric("precision", precision_macro)
    vv.add_metric("recall", recall_macro)

    vv.validate()


def validate_2():
    dataset = MyDataset(process_type="testing", reshape_size=(224,224))
    dataloader = DataLoader(dataset, 1200, True)

    vv = ValidateVessel(dataloader, ResNet50())

    vv.gpu()

    vv.load_model_para("resnet.pth")
    vv.set_tensorboard_dir("./logger/resnet50_split_val")
    vv.add_metric("precision", precision_macro)
    vv.add_metric("recall", recall_macro)

    vv.validate()


validate_2()
