import torch.nn as nn
import torch
import time
import numpy as np
import cv2
from constrained_conv import ConstrainedConv2d
from fan import FAN
from dxtorchutils.ImageClassification.models import ResNet18


def aaa():
    loss_func = torch.nn.CrossEntropyLoss()

    raw = cv2.imread("/Users/iandx/Documents/Documents/Files/DeepfakeDetection/IrisSegmentation/src/resources/data/training/iris_raw/C1_S1_I1.tiff")
    raw = raw.transpose(2, 0, 1) // 225
    label = cv2.imread("/Users/iandx/Documents/Documents/Files/DeepfakeDetection/IrisSegmentation/src/resources/data/training/iris_ground_truth/C1_S1_I1_gt.png", 0)
    label = label // 255

    data = torch.unsqueeze(torch.from_numpy(raw), 0).type(torch.FloatTensor)
    targets = torch.unsqueeze(torch.from_numpy(label), 0).type(torch.LongTensor)
    cc = ConstrainedConv2d(3, 2, 5, padding=2, scaling_rate=1)
    opt = torch.optim.SGD(cc.parameters(), 0.0005, 0.01)

    for i in range(1000):
        output = cc(data)
        loss = loss_func(output, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()


        # print(cc.weight)

        print(loss)
    print(cc.weight)

    print(output.data.numpy().shape)
    prediction = torch.max(output, 1)[1].type(torch.LongTensor).data.numpy()
    prediction = (np.reshape(prediction, (300, 400)) * 255).astype(np.uint8)

    cv2.imshow("new", prediction)
    print(np.where(label != 0))
    print(np.where(prediction != 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_():
    fan = FAN(2)
    fan.cuda()
    fan.load_state_dict(torch.load("models/fan.pth"))

    print(fan.constrained_layer.weight)


filter_()