import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from neural_process import NeuralProcess, NeuralProcessImg, NP
# from datasets import SineData
from math import pi
from torch.utils.data import DataLoader
from training import NeuralProcessTrainer, NPTrainer
from torch.utils.data import DataLoader
from create_dataset import FFTDataset
import imageio
from torchvision.utils import make_grid
from utils import inpaint







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Instantiate the FFTDataset
root = './data'
dataset = FFTDataset(root, train=True, download=True)

# Create a DataLoader
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fetch the first two batches and print their shapes
# for i, ((magnitude, phase, img), labels) in enumerate(train_loader):
#     # if i >= 2:
#     #     break
#     print(f"Batch {i+1}:")
#     print("Magnitude shape: ", torch.max(magnitude))
#     # print("Phase shape: ", torch.max(phase) )
#     # print("Labels shape: ", labels.shape)
#     # print("img shape: ", img.shape)
#     # print(magnitude.shape)








config = {
  "img_size": [3, 32, 32],
  "batch_size": 16,
  "r_dim": 1000,
  "h_dim": 1000,
  "z_dim": 1000,
  "num_context_range": [1, 200],
  "num_extra_target_range": [0, 200],
  "lr": 1e-3,
  "epochs": 5,
  "dataset": "celeba"
}

img_size = config["img_size"]
batch_size = config["batch_size"]
r_dim = config["r_dim"]
h_dim = config["h_dim"]
z_dim = config["z_dim"]
num_context_range = config["num_context_range"]
num_extra_target_range = config["num_extra_target_range"]
epochs = config["epochs"]


np_img = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)

optimizer = torch.optim.Adam(np_img.parameters(), lr=config["lr"])

np_trainer = NeuralProcessTrainer(device, np_img, optimizer,
                                  num_context_range, num_extra_target_range,
                                  print_freq=100)


for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    np_trainer.train(train_loader, 1)







batch_size = 2
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for i, ((magnitude, phase, img), labels) in enumerate(test_loader):
    if i >= 1:
        break
    all_imgs = magnitude
    img = img
    for j in range(magnitude.shape[0]):

        all_imgs[j] = img[j] #torch.Tensor(img[j].transpose(2, 0, 1) / 255.)
    img_grid = make_grid(all_imgs, nrow=2, pad_value=1.)
    plt.imshow(img_grid.permute(1, 2, 0).numpy())
    plt.show()
    context_mask = (torch.Tensor(32, 32).uniform_() < .9).byte()
    context_mask = 1-context_mask
    all_inpaintings = torch.zeros(2, 3, 32, 32)
    for j in range(all_inpaintings.shape[0]):
        all_inpaintings[j] = inpaint(np_img, img[0], context_mask, device)
    img_grid = make_grid(all_inpaintings, nrow=2, pad_value=1.)
    plt.imshow(img_grid.permute(1, 2, 0).numpy())
    plt.show()


