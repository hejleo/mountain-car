import torch
import torchvision
import torchvision.transforms as transforms
from dataload_classifier import Dataset
import torchvision.models as models
import IPython
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

batch_size = 4
inp = torch.zeros((4, 3, 128, 128), dtype=torch.float32)



# ------------------------ Model & Dataset setup -------------------------------
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 2)

train_set = Dataset(img_dir='/home/lravaglia/work/openai-gym/dataset/train',
    annotations_file='/home/lravaglia/work/openai-gym/dataset/train-labels.csv')
test_set = Dataset(img_dir='/home/lravaglia/work/openai-gym/dataset/test',
    annotations_file='/home/lravaglia/work/openai-gym/dataset/test-labels.csv')

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=True, num_workers=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



# ------------------------------- Training -------------------------------------
model.train()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        optimizer.zero_grad()

        # Normalize inputs - from Imagenet pretraining
        inp = (inputs).float()
        inp[:, 0, :, :] = ((inp[:, 0, :, :] - 0.485) / 0.229)
        inp[:, 1, :, :] = ((inp[:, 1, :, :] - 0.456) / 0.224)
        inp[:, 2, :, :] = ((inp[:, 2, :, :] - 0.406) / 0.225)

        # forward + backward + optimize
        outputs = model(inp)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        IPython.embed()

    if epoch%4==0:
        optim = torch.optim.SGD(model.parameters(), lr=0.01*1/(10**(epoch/4)))


# Saving/Loading model parameters after training
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))



# -------------------------------- Testing -------------------------------------
model.eval()
corr_cnt = 0
tot_cnt = 0

for i, data in enumerate(testloader, 0):

    inputs, labels = data

    inp = (inputs).float()

    inp[:, 0, :, :] = ((inp[:, 0, :, :] - 0.485) / 0.229)
    inp[:, 1, :, :] = ((inp[:, 1, :, :] - 0.456) / 0.224)
    inp[:, 2, :, :] = ((inp[:, 2, :, :] - 0.406) / 0.225)

    outputs = model(inp)

    if outputs[0][0].item()>outputs[0][1].item():
        if labels.item()==0:
            corr_cnt += 1
    else:
        if labels.item()==1:
            corr_cnt += 1
    tot_cnt +=1
    print(inp.shape)

print("Test accuracy: ", corr_cnt/tot_cnt)
print('Finished Training')
