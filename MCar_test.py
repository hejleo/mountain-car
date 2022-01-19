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

# --------------------------------- Setup --------------------------------------
# Define videocapture
vid = cv2.VideoCapture(0)

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Import model and load with trained parameters
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

model_input = torch.zeros((1, 3, 128, 128), dtype=torch.float32)



# --------------------------------- Utils --------------------------------------
def img_process(frm, mod_inp):
    """
    Since cam frames are shaped (480,640), we crop and resize them to fit the
    required image size. Additionally, we normalize for the statistics collected
    on the Imagenet pretraining.
        Args:
            frm (numpy.ndarray): frame acquired by the camera.
            mod_inp: dummy model input.
        Returns:
            input: the input tensor to the model
    """

    # Crop
    frame_tens = ToTensor()(frm)
    frame_tens = frame_tens[:,:,80:560]

    # Resize
    input = F.interpolate(frame_tens, size =128)
    input = input.permute(0, 2, 1)
    input = F.interpolate(input, size=128)
    input = input.permute(0, 2, 1)

    # Normalize
    input = input[None, :, :, :]
    input[0, 0, :, :] = ((input[0, 0, :, :] - 0.485) / 0.229)
    input[0, 1, :, :] = ((input[0, 1, :, :] - 0.456) / 0.224)
    input[0, 2, :, :] = ((input[0, 2, :, :] - 0.406) / 0.225)

    return input



# ----------------------------- Model Inference --------------------------------
for step in range(10000):

    env.render()

    # camera image processing
    ret, frame = vid.read()

    model_input =img_process(frame, model_input)

    outputs = model(model_input)

    if outputs[0][0].item()>outputs[0][1].item():
        car_action = 0
        print("pushing Left")
    else:
        car_action = 2
        print("pushing Right")

    out = env.step(car_action) # 2 to move right, 0 to move left
    if out[2]==True:
        print("Finished in {} step!".format(step))
        break

env.close()
