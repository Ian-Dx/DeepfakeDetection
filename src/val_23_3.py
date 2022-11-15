# from dxtorchutils.ImageClassification.validate import ValidateVessel
from torch.utils.data import DataLoader
from fan import FAN
from dataset import MyDataset
from dxtorchutils.utils.metrics import precision_macro, recall_macro
from tesing import ValidateVessel
from dxtorchutils.ImageClassification.models import ResNet50

from noise_net import NoiseNet
def validate_():
    dataset = MyDataset(process_type="testing", reshape_size=(224,224),dataset_type="c23")
    dataloader = DataLoader(dataset, 1200, True)

    vv = ValidateVessel(dataloader, NoiseNet())

    vv.gpu()

    vv.load_model_para("models/filter2.pth")
    # vv.set_tensorboard_dir("./logger/filter2_val_0")

    vv.validate()


validate_()
