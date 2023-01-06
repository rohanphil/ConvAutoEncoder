import torch
from dataloaders import *
import numpy as np 
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
	def __init__(self, latent_dim):

		super().__init__()

		self.encoder_cnn = nn.Sequential(
			nn.Conv2d(1,8,3,stride = 2, padding = 1),
			nn.ReLU(),
			nn.Conv2d(8,16,3, stride = 2, padding = 1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(16,32,3,stride=2,padding = 0),
			nn.ReLU()
			)

		self.flatten = nn.Flatten(start_dim = 1)

		self.linear_enc = nn.Sequential(
			nn.Linear(3*3*32, 128),
			nn.ReLU(),
			nn.Linear(128,latent_dim)
			) 

	def forward(self,x):
		x = self.encoder_cnn(x)
		x = self.flatten(x)
		return self.linear_enc(x)


class Decoder(nn.Module):

	def __init__(self,latent_dim):

		super().__init__()

		self.linear_dec = nn.Sequential(
			nn.Linear(latent_dim,128),
			nn.ReLU(),
			nn.Linear(128,3*3*32),
			nn.ReLU()
			)

		self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (32,3,3))

		self.decoder_conv = nn.Sequential(
			nn.ConvTranspose2d(32,16,3,stride = 2, output_padding = 0),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.ConvTranspose2d(16,8,3,stride = 2, padding = 1, output_padding = 1),
			nn.BatchNorm2d(8),
			nn.ReLU(),
			nn.ConvTranspose2d(8,1,3,stride = 2, padding = 1, output_padding = 1)
			)

	def forward(self, x):
		x = self.linear_dec(x)
		x = self.unflatten(x)
		x = self.decoder_conv(x)
		x = torch.sigmoid(x)
		return x


if __name__ == "__main__":

	Encoder = Encoder(500)
	Decoder = Decoder(500)

	print(f"Encoder : {Encoder}")
	print(f"Decoder : {Decoder}")