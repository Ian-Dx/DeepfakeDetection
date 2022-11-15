import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, fake_type="Deepfakes", dataset_type="c0", reshape_size=(256, 256), process_type="training"):
        dir = "/home/lzy/lizuoyan/data/FFpp-faces/"
        self.reshape_size = reshape_size
        self.fake_dir = dir + fake_type + "/" + dataset_type + "/images/"

        self.real_dir = dir + "Origin/" + dataset_type + "/images/"
        self.paths = []
        self.labels = []

        # idx = 0

        for file in os.listdir(self.real_dir):
            video_path = self.real_dir + file
            if process_type == "training":
                if os.path.isdir(video_path) and int(file[:3]) < 990:
                    for file0 in os.listdir(video_path):
                        if file0.endswith(".jpg"):
                            self.paths.append(video_path + "/" + file0)
                            self.labels.append(1)
            else:
                if os.path.isdir(video_path) and int(file[:3]) >= 990:
                    for file0 in os.listdir(video_path):
                        if file0.endswith(".jpg"):
                            self.paths.append(video_path + "/" + file0)
                            self.labels.append(1)


        for file in os.listdir(self.fake_dir):
            video_path = self.fake_dir + file
            if process_type == "training":
                if os.path.isdir(video_path) and int(file[:3]) < 990:
                    for file0 in os.listdir(video_path):
                        if file0.endswith(".jpg"):
                            self.paths.append(video_path + "/" + file0)
                            self.labels.append(0)

            else:
                if os.path.isdir(video_path) and int(file[:3]) >= 990:
                    for file0 in os.listdir(video_path):
                        if file0.endswith(".jpg"):
                            self.paths.append(video_path + "/" + file0)
                            self.labels.append(0)


        print(len(self.paths))


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], 0)
        # print(img.shape)
        img = cv2.resize(img, self.reshape_size)
        # img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.unsqueeze((torch.from_numpy(img) / 255).type(torch.FloatTensor), 0)

        return img_tensor, torch.scalar_tensor(self.labels[index]).type(torch.LongTensor)
