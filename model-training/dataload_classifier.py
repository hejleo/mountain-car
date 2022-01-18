# Dataloader - taken from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import pandas as pd
import IPython
from PIL import Image
from torchvision import transforms

class Dataset(object):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]+'.jpg')

        img = Image.open(img_path)
        convert_tensor = transforms.ToTensor()
        image = convert_tensor(img)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":

    dataset = Dataset(img_dir='/home/lravaglia/work/continual-AI-deck/dataset-cropped/train', annotations_file='/home/lravaglia/work/continual-AI-deck/labels-train.csv')
