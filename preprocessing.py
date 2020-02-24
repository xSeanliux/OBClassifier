import csv 
import numpy as np
import pickle
import pdb
import torch
from sklearn.preprocessing import normalize

PATH = "./OB統整_正規化.csv" #TODO
SAVE_PATH = "./data.pkl"
input_columns = 14



with open(PATH) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	neg = 0
	pos = 0
	all_data = []	

	for row in csv_reader:
		current_data = dict()
		inputs = []
		isValid = True;
		for c in range(0, input_columns):
			try:
				inputs.append(float(row[c]))
			except ValueError:
				isValid = False
				break
		if isValid == False:
			continue

		print(inputs)
		inputs = np.array(inputs)
		
		inputs = normalize(inputs[:, np.newaxis], axis = 0) #normalize
		inputs = torch.from_numpy(inputs).squeeze()
		current_data["input"] = inputs 
		current_data["output"] = int(row[input_columns])
		if(row[input_columns] == "1"):
			pos = pos + 1
		else:
			neg = neg + 1
		all_data.append(current_data)
	print("Finished preprocessing")
	with open(SAVE_PATH, "wb") as f:
		pickle.dump(all_data, f)

print("Pos: " + str(pos) + ", Neg: " + str(neg))
