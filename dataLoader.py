import pickle 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

class starDatas(Dataset):
	arr = []
	def __init__(self):
		self.arr = pickle.load(open("data.pkl", "rb"))
	def __getitem__(self, index):
		#Check if it's an OB star or not
		return self.arr[index]

	def __len__(self):
		return len(self.arr)
