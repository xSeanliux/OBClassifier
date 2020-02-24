import numpy as np 
import torch 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import pdb
from tqdm import tqdm 
import dataLoader as data
from torch.utils.tensorboard import SummaryWriter

PATH = "./training_weights.ckpt"
device = "cpu"

learning_rate = 0.0001
batch_size = 32

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(14, 256) #input layer 
		self.fc2 = nn.Linear(256, 256) #hidden layer
		self.out = nn.Linear(256, 2) #output layer
	
	def forward(self, x):
		x = x.double()
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.out(x)
		return x
	

