import numpy as np 
import torch
from utils import *
from dataloaders import *
from model import *



def train_epoch(encoder, decoder, device, dataloader,loss, opt):
	encoder.train()
	decoder.train()
	train_loss = []

	for batch,_ in dataloader:

		batch = batch.to(device)
		encoded_data = encoder(batch)
		decoded_data = decoder(encoded_data)
		criterion = loss(decoded_data, batch)
		opt.zero_grad()
		criterion.backward()
		opt.step()
		#print(f"component loss : {criterion.data}")
		train_loss.append(criterion.detach().cpu().numpy())

	return np.mean(train_loss)


def train_ae(encoder, decoder, train_set, batch_size, optim, device, loss, num_epochs, save_model = False):
	train_loader = dataloader(batch_size = batch_size).create_loader(train_set)
	train_loss_track = []
	for i in range(num_epochs):
		train_loss = train_epoch(encoder,decoder,device, train_loader, loss, optim)
		train_loss_track.append(train_loss)
		print(f"training loss at {i} epoch = {train_loss}")
	if save_model:
		torch.save(encoder,"encoder.pt")
		torch.save(decoder,"decoder.pt")
		torch.save(train_loss_track,"train_loss.pt")

	return (encoder,decoder, train_loss_track)



if __name__ == "__main__":
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

	train, test = Dataset("data").load()
	train = Transform().transform(train)
	test = Transform().transform(test)

	train_loader = dataloader(batch_size = 100).create_loader(train)
	test_loader = dataloader(batch_size = 100).create_loader(test)


	num_epochs = 2

	import tqdm

	for i in tqdm.tqdm(range(num_epochs)):
		train_loss = train_epoch(encoder,decoder,device, train_loader, loss, optim)
		print(f"training loss at {i} epoch = {train_loss}")



