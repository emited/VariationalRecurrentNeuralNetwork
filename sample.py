import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN

#hyperparameters
x_dim = 28
h_dim = 100
z_dim = 16
n_layers =  1

state_dict = torch.load('saves/vrnn_state_dict_41.pth')
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)

sample = model.sample(28*6)
plt.imshow(sample.numpy(), cmap='gray')
plt.show()