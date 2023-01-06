import torch
import numpy as np
import pandas as pd 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split



class Dataset(object):
	def __init__(self, datapath):
		self.train = []
		self.test = []
		self.path = datapath 

	def load(self,download = True):

		self.train = torchvision.datasets.MNIST(self.path, train=True, download=download)
		self.test  = torchvision.datasets.MNIST(self.path, train=False, download=download)
		return (self.train, self.test)

class Transform(object):

	def __init__(self):
		self.transforms = transforms.Compose([transforms.ToTensor()])

	def transform(self,dataset):
		dataset.transform = self.transforms
		return dataset

class CreateLoss(object):
	def __init__(self):
		self.loss = torch.nn.MSELoss

if __name__ == "__main__":

	#Basic tests

	#test to check if loaded
	def load_test():
		train, test = [],[]
		train, test = Dataset("data").load()
		try:
			assert(train !=[])
			print("Dataset Loaded!")
		except:
			print("Dataset Failed to load")

	def test_transform():
		train, test = Dataset("data").load()
		try:
			assert(type(Transform().transform(train)[0][0])==torch.Tensor)
			print("Transformation successful")
		except:
			print("ERROR: Transformation not successful")

	def test_loss():
		x = torch.Tensor([1,1,1])
		x.requires_grad = True
		y = torch.Tensor([1,1,1])
		loss = CreateLoss().loss()

		try:
			loss = CreateLoss().loss()
			output = loss(x,y)
			assert(output.item() == 0)
			print("Loss works and is backpropable")
		except:
			print("ERROR in loss function")





	from argparse import ArgumentParser
	parser = ArgumentParser(add_help=True)

	parser.add_argument("--load_test", action='store_true',
						help="Test to see if the dataset loads")
	parser.add_argument("--test_transform", action='store_true',
						help="Test to see if the dataset is transformed to a tensor")
	parser.add_argument("--test_loss", action='store_true',
						help="Test to see if the the loss is working as intended")

	args = parser.parse_args()

	if args.load_test:
		load_test()
	if args.test_transform:
		test_transform()
	if args.test_loss:
		test_loss()









