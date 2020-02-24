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


net = Net().train().to(device)
net = net.double()

learning_rate = 0.01

writer = SummaryWriter()
g_checkpoint = torch.load("./training_weights.ckpt", map_location = torch.device(device)) #the file to train
net.load_state_dict(g_checkpoint['model'])
#Will train from the same file every time, if you don't have yet make sure to just comment     this out
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.5) #Not sure what the parameters do    , just copying it
optimizer.load_state_dict(g_checkpoint['optimizer'])

criterion = nn.CrossEntropyLoss() 

def train(epochs):
	print("Begin loading")
	datas = data.starDatas()
	print("Done")
	dataset = torch.utils.data.DataLoader(datas, batch_size = batch_size, shuffle = True)
	print("Dataset:")
	try:
		total_it = 0
		for epoch in range(epochs):
			running_loss = 0
			iters = 0
			for i, datai in enumerate(tqdm(dataset)):
				total_it = total_it + 1
				iters = iters + 1
				output = net(datai["input"])
				optimizer.zero_grad()
				loss = criterion(output, datai["output"])
				#aaa = list(net.parameters())[0].clone() 
				loss.backward()
				optimizer.step()
				#bbb = list(net.parameters())[0].clone()
				#pdb.set_trace()
				running_loss = running_loss + loss.item()
				if (epoch % 1000 == 999):
					torch.save({
					"epoch": epoch,
					"model": net.state_dict(),
					"optimizer": optimizer.state_dict()
					}, "./save_epoch_{}.ckpt".format(epoch))
					print("Saving on epoch " + str(epoch))
			writer.add_scalar("Loss", running_loss/iters, total_it)
		
	except KeyboardInterrupt: 
		print("Saving on exit")
		torch.save({
        "epoch": epoch,
		"model": net.state_dict(),
		"optimizer": optimizer.state_dict()
		}, PATH)
 
train(10000)