# Importing Image class from PIL module
import PIL
from PIL import Image
import psutil
import time
from torchvision import transforms
import glob, os
import csv
import torchvision.transforms.functional as fn


for file in glob.glob("/home/pulp/openai-gym/dataset/train/"+"*.jpg"):

    im = Image.open(file)
    im = fn.center_crop(im, output_size=[480])
    im = fn.resize(im, size=[128])
    width, height = im.size
    print(width, height)
    #im.save(file)
