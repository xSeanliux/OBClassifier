import numpy as np 
from model import Net
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

device = "cpu"

datas = data.starDatas()
dataset = torch.utils.data.DataLoader(datas, shuffle = True)

net = Net().double()
g_checkpoint = torch.load("./training_weights.ckpt", map_location = torch.device(device)) #the file to train
net.load_state_dict(g_checkpoint['model'])

correct = 0
tot = 0
tot_0 = 0
tot_1 = 0
cor_0 = 0
cor_1 = 0

for i, data in enumerate(tqdm(dataset)):
	tot = tot + 1
	res = net(data["input"])
	_, pred = torch.max(res, 1)
	if pred == data["output"]:
		correct = correct + 1
		if pred[0] == 0:
			cor_0 = cor_0 + 1
		else:
			cor_1 = cor_1 + 1
	
	if data["output"] == 0:
		tot_0 = tot_0 + 1
	else:
		tot_1 = tot_1 + 1

print("Total: {}, Correct: {}, Rate: {}%".format(tot, correct, correct / tot * 100))
print("Total 1: {}, Correct: {}, Rate: {}%".format(tot_1, cor_1, cor_1 / tot_1 * 100))
print("Total 0: {}, Correct: {}, Rate: {}%".format(tot_0, cor_0, cor_0 / tot_0 * 100))