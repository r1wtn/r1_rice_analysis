import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

from random import seed

class CountryClusterRiceDataset(Dataset):
    def __init__(self, root, input_resolution=224, mode="train", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        root (str): データセットのルートフォルダ
        """
        self.input_resolution = input_resolution
        self.mean = mean
        self.std = std

        root = Path(root)
        if not root.exists():
            print("root folder does not exist.")
            exit()
        # 国名数
        self.class_num = len(list(root.iterdir()))

        self.target_image_list = []
        self.target_label_list = []

        for i, c in enumerate(root.iterdir()):
            df = pd.read_csv(c.joinpath("data.csv"))
            df = df[df["trainval"]==mode]
            self.target_image_list += list(df["local_image_path"])
            self.target_label_list += [i] * len(df)


    def __len__(self):
        return len(self.target_image_list)

    def convert_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(
            image, (self.input_resolution, self.input_resolution))  # 512x512

        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.mean).astype(np.float32)) / \
            np.array(self.std).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, index):
        target_image_path = self.target_image_list[index]
        label = self.target_label_list[index]
        image = self.convert_image(target_image_path)
        return image, label