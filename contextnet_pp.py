import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, dilation=1, padding=0):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride, dilation=dilation,padding=padding)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		return x


class idxLayer(torch.nn.Module):
	def __init__(self):
		super(idxLayer, self).__init__()

	def forward(self, x):
		num, c, h, w = x.size()
		cell=h/12
		for i in range(12):
			k=int(i*cell)
			t=x[:,:,k,:]
			if i==0:
				tensor=t
			else:
				tensor=torch.cat((tensor,t),2)
		return tensor

class Context(nn.Module):
	def __init__(self):
		super(Context, self).__init__()
		self.con1=BasicConv2d(21,  2048, [3,1],1,padding=[1,0])
		self.con2=BasicConv2d(2048,1024, [3,1],1,padding=[1,0])
		self.con3=BasicConv2d(1024, 512, [3,1],1,padding=[1,0])
		self.con4=BasicConv2d(512,  256, [3,1],1,padding=[1,0])
		self.con5=BasicConv2d(256,  128, [3,1],1,dilation=2,padding=[2,0])
		self.con6=BasicConv2d(128,   64, [3,1],1,dilation=4,padding=[4,0])
		self.con7=BasicConv2d(64,    32, [3,1],1,dilation=8,padding=[8,0])
		self.con8=BasicConv2d(32,    16, [3,1],1,dilation=16,padding=[16,0])
		self.con10=BasicConv2d(4080,640, [3,1],1,padding=[1,0])
		self.con11=BasicConv2d( 640,  8, [3,1],1,padding=[1,0])
		self.con12=BasicConv2d(   8,  8, [3,1],1,padding=[1,0])
		self.con13=BasicConv2d(   8,  8, [3,1],1,padding=[1,0])
		self.con14=BasicConv2d(   8,  8, [3,1],1,padding=[1,0])
		self.con16=BasicConv2d( 32,  32, [3,1],1,padding=[1,0])
		self.con18=BasicConv2d(693, 320, [1,1],1,padding=[0,0])
		self.idxl=idxLayer()
		self.first_linear = nn.Linear(3840, 512)

	def forward(self, x):
		x1=self.con1(x)
		x2=self.con2(x1)
		x3=self.con3(x2)
		x4=self.con4(x3)
		x5=self.con5(x4)
		x6=self.con6(x5)
		x7=self.con7(x6)
		x8=self.con8(x7)
		x9=torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
		x10=self.con10(x9)
		x11=self.con11(x10)
		x12=self.con12(x11)
		x13=self.con13(x12)
		x14=self.con14(x13)
		x15=torch.cat((x11,x12,x13,x14),1)
		x16=self.con16(x15)
		x17=torch.cat((x,x10,x16),1)
		x18=self.con18(x17)
		x19=self.idxl(x18)
		x=x19.view(x19.size(0), -1)
		x=self.first_linear(x)
		return x


class PP(nn.Module):
	def __init__(self):
		super(PP, self).__init__()
		self.context=Context()
		self.l1 = nn.Linear(1024, 1024)
		self.relu= nn.ReLU()
		self.l2 = nn.Linear(1024, 2)

	def forward(self, x1,x2):
		x1=self.context(x1)
		x2=self.context(x2)
		x=torch.cat((x1,x2),1)
		x=self.relu(self.l1(x))
		x=self.l2(x)
		return x
