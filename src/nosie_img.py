import cv2
import torch
from fan import FAN
import numpy as np
import os
from noise_net import NoiseNet
import torch.nn.functional as F


def img_to_tensor(img):
    img = cv2.resize(img, (256, 256)) / 255
    # img = np.transpose(img, (2,0,1))
    tensor_img = torch.from_numpy(img)
    tensor_img = torch.unsqueeze(torch.unsqueeze(tensor_img, 0), 0).type(torch.FloatTensor)
    tensor_img = tensor_img.cuda()

    return tensor_img


def fan_(num):
    fake = cv2.imread("/home/lzy/lizuoyan/data/FFpp-faces/Deepfakes/c0/images/001_870/{}.jpg".format(num), 0)
    real = cv2.imread("/home/lzy/lizuoyan/data/FFpp-faces/Origin/c0/images/001/{}.jpg".format(num), 0)


    fake_tensor = img_to_tensor(fake)

    real_tensor = img_to_tensor(real)

    fan = FAN(2)
    fan.cuda()
    fan.load_state_dict(torch.load("models/fan4.pth"))

    fake_output = fan.constrained_layer(fake_tensor)
    fake_output = torch.squeeze(fake_output).cpu().data.numpy()[0]
    fake_output = (fake_output * 255).astype(np.uint8)

    real_output = fan.constrained_layer(real_tensor)
    real_output = torch.squeeze(real_output).cpu().data.numpy()[0]
    real_output = (real_output * 255).astype(np.uint8)

    cv2.imwrite("outputs_fan/fake000/fake{}.png".format(num), fake_output)
    cv2.imwrite("outputs_fan/real000/real{}.png".format(num), real_output)


def noise_net_forward(model, input):
    h, w = input.shape[-2:]
    # (n, 3, 224, 224)
    x = model.down_sample(input)
    # (n, 64, 112, 112)
    x = model.conv(x)
    # (n, 128, 112, 112)
    x = F.interpolate(x, (h, w), None, "bilinear", True)
    # (n, 128, 224, 224)
    x = model.up_sample(x)
    # (n, 3, 224, 224)
    output = input - x
    # (n, 3, 224, 224)

    return output

def noisenet_(num):
    fake = cv2.imread("/home/lzy/lizuoyan/data/FFpp-faces/Deepfakes/c0/images/000_003/{}.jpg".format(num))
    real = cv2.imread("/home/lzy/lizuoyan/data/FFpp-faces/Origin/c0/images/000/{}.jpg".format(num))


    fake_tensor = img_to_tensor(fake)
    real_tensor = img_to_tensor(real)

    noise_net = NoiseNet()
    noise_net.cuda()
    noise_net.load_state_dict(torch.load("models/noisenet_lite.pth"))

    fake_output = noise_net_forward(noise_net, fake_tensor)
    fake_output = torch.squeeze(fake_output).cpu().data.numpy()
    fake_output = (np.transpose(fake_output, (1,2,0)) * 255).astype(np.uint8)

    real_output = noise_net_forward(noise_net, real_tensor)
    real_output = torch.squeeze(real_output).cpu().data.numpy()
    real_output = (np.transpose(real_output, (1,2,0)) * 255).astype(np.uint8)

    cv2.imwrite("outputs_noisenet/fake000/fake_out_{}.png".format(num), fake_output)
    cv2.imwrite("outputs_noisenet/real000/real_out_{}.png".format(num), real_output)

idx = 0

# os.mkdir("outputs_fan")
# #
# os.mkdir("outputs_fan/real000")
# os.mkdir("outputs_fan/fake000")
for a in os.listdir("/home/lzy/lizuoyan/data/FFpp-faces/Deepfakes/c0/images/001_870/"):
    fan_(a[:3])
    if idx == 5:
        break
    print(idx)
    idx += 1