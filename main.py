import numpy as np 
import torch
from utils import *
from dataloaders import *
from model import *
from train_autoencoder import *
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

class LinearHead(nn.Module):

	def __init__(self,input_dim):
		super().__init__()
		self.linearhead = nn.Sequential(
			nn.Linear(input_dim, 32),
			nn.ReLU(),
			nn.Linear(32, 10)
			)
		self.softmax = nn.Softmax(dim = -1)

	def forward(self,x):
		return self.linearhead(x)


test = True


encoder = Encoder(100)
decoder = Decoder(100)

loss = CreateLoss().loss()

device = get_device().device

params_opt = [
	{'params' : encoder.parameters()},
	{'params' : decoder.parameters()}
]

optim = torch.optim.Adam(params_opt, lr = 0.001,weight_decay = 1e-05)

encoder.to(device)
decoder.to(device)
print(device)

#print(encoder.state_dict())

train, test = Dataset("data").load()
train = Transform().transform(train)
test = Transform().transform(test)
tl = []
test_loss = []
try:
	encoder = torch.load("encoder.pt")
	decoder = torch.load("decoder.pt")
	tl = torch.load("train_loss.pt")
	if test:
		test_loss = torch.load("test_loss.pt")
except:
	if test:
		encoder, decoder, tl, test_loss = train_ae(encoder, decoder, train, 100, optim, device, loss, 30, save_model = True, test = True, test_set = test)
	else:

		encoder, decoder, tl = train_ae(encoder, decoder, train, 100, optim, device, loss, 30, save_model = True)
#print(tl)

def train_linearhead(encoder,device,train, test, optim, batch_size = 100, num_epochs = 10):

	train_loader = dataloader(batch_size = batch_size).create_loader(train)
	encoder.eval()
	linear_model = LinearHead(100).to(device)
	optim = torch.optim.Adam(linear_model.parameters(), lr = 0.01,weight_decay = 1e-02)
	loss = nn.CrossEntropyLoss()

	for n in range(num_epochs):
		lin_train = []

		for batch,labels in train_loader:
			batch = batch.to(device)
			encoded_data = encoder(batch).detach()
			output = linear_model(encoded_data)

			#print(type(output[0].item()))
			#print(output)
			l = loss(output.cpu(), labels)
			optim.zero_grad()
			l.backward()
			optim.step()
			lin_train.append(l.data)
		print(f"linear training loss at epoch {n} = {np.mean(lin_train)}")

	return (encoder, linear_model)

encoder, linear_model = train_linearhead(encoder, device,train, test, optim)


def test_acc(encoder,linear_model, test_set, batch_size):
	acc = 0
	n = len(test_set)

	encoder.eval()
	linear_model.eval()

	test_loader = dataloader(batch_size = batch_size).create_loader(test_set)

	for batch,label in test_loader:
		batch = batch.to(device)
		label = label.to(device)
		encoded_data = encoder(batch).detach()
		output = torch.argmax(linear_model(encoded_data), dim = -1)
		acc += len(torch.where(output == label)[0])
		#print(acc)
	return acc/n



test_acc = test_acc(encoder, linear_model, test, batch_size = 100)
print(f"The final test accuracy = {test_acc}")



if __name__ == "__main__":

	from argparse import ArgumentParser
	parser = ArgumentParser(add_help=True)

	parser.add_argument("--plot_loss", action='store_true',
						help="plot_loss")

	args = parser.parse_args()

	if args.plot_loss and tl:
		plt.plot(tl, label = "training loss")
		#print(test_loss)
		if test_loss:
			plt.plot(test_loss, label = "testing loss")
		plt.legend()
		plt.savefig("Plots/AElosses.png")
		plt.show()





