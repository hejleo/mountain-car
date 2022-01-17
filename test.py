import numpy as np
import gym
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Listener
import keyboard
import cv2
import torch
import torchvision.models as models
import torchvision.transforms.functional as fn
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage

# define videocapture
vid = cv2.VideoCapture(0)

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')

# import model and load with trained parameters
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

model_inp = torch.zeros((1, 3, 128, 128), dtype=torch.float32)

env.reset()

for _ in range(10000):

    env.render()

    # camera image processing
    ret, frame = vid.read()
    frame_tens = ToTensor()(frame)
    frame_tens = frame_tens[:,:,80:560]
    input = F.interpolate(frame_tens, size =128)  #The resize operation on tensor.
    input = input.permute(0, 2, 1)
    input = F.interpolate(input, size=128)
    input = input.permute(0, 2, 1)

    model_inp[0, 0, :, :] = ((input[0, :, :] - 0.485) / 0.229)
    model_inp[0, 1, :, :] = ((input[1, :, :] - 0.456) / 0.224)
    model_inp[0, 2, :, :] = ((input[2, :, :] - 0.406) / 0.225)

    # model inference
    outputs = model(model_inp)

    if outputs[0][0].item()>outputs[0][1].item():
        car_input = 0
        print("left")
    else:
        car_input = 2
        print("right")

    env.step(car_input) # 2 to right, 0 to left

env.close()
