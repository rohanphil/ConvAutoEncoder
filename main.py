import numpy as np 
import torch
from utils import *
from dataloaders import *
from model import *
from train_autoencoder import *

import matplotlib.pyplot as plt

test = True


encoder = Encoder(8)
decoder = Decoder(8)

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





