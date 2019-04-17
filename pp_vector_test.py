from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
import pretrainedmodels
import copy
from PIL import Image
import contextnet_pp
from sklearn import metrics
from sklearn.model_selection import KFold


test=[]
for line in open('test.txt'):
	line=line.strip()
	line=line.split()
	test.append([line[0],line[1],line[2]])



seqdic={'A':0, 'R':1, 'D':2, 'C':3, 'Q':4, 'E':5, 'H':6, 'I':7, 'G':8, 'N':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}


def g_data_test(n):
	l1=len(test[n][0])
	l2=len(test[n][1])
	data1=test[n][0]
	data2=test[n][1]
	x1=np.zeros([21,l1],dtype=np.float32)
	x2=np.zeros([21,l2],dtype=np.float32)
	y=np.zeros([1],dtype=np.float32)
	for i in range(len(data1)):
		if data1[i] in seqdic:
			x1[seqdic[data1[i]]][i]=1.0
		else:
			x1[20][i]=1.0
	for i in range(len(data2)):
		if data2[i] in seqdic:
			x2[seqdic[data2[i]]][i]=1.0
		else:
			x2[20][i]=1.0

	x1=x1.reshape([1,21,l1,1])
	x2=x2.reshape([1,21,l2,1])
	x1=torch.from_numpy(x1)
	x2=torch.from_numpy(x2)
	y[0]=int(test[n][2])
	y=torch.from_numpy(y)
	y=y.long()
	return x1,x2,y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = contextnet_pp.PP()

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for cv in range(10):
	pth_file='./model/context'+str(cv+1)+'.pth'
	model.load_state_dict(torch.load(pth_file))
	model.eval()
	running_corrects = 0
	k=0
	for i in range(len(test)):
		inputs1,inputs2,labels=g_data_test(i)
		inputs1 = inputs1.to(device)
		inputs2 = inputs2.to(device)
		labels = labels.to(device)
		outputs = model(inputs1,inputs2)
		_, preds = torch.max(outputs, 1)
		running_corrects += torch.sum(preds == labels.data)
		k+=1

	test_acc = running_corrects.double() /k
	print('test:',cv+1,test_acc)
