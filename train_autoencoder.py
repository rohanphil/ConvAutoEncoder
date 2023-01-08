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


def test_epoch(encoder,decoder, device, dataloader, loss):
	
	encoder.eval()
	decoder.eval()

	with torch.no_grad():

		output = []
		label = []

		for label_image,_ in dataloader:

			images = label_image.to(device)
			encode = encoder(images)
			decode = decoder(encode)
			output.append(decode.cpu())
			label.append(images.cpu())

		output = torch.cat(output)
		label = torch.cat(label)

		l = loss(output,label)

	return l.data


def train_ae(encoder, decoder, train_set, batch_size, optim, device, loss, num_epochs, save_model = False, test = False, test_set = None):
	train_loader = dataloader(batch_size = batch_size).create_loader(train_set)
	train_loss_track = []
	test_loss_track = []
	if test:
		try:
			assert(test_set != None)
			test_loader = dataloader(batch_size = batch_size).create_loader(test_set)
		except:
			print("Please Specify test set")
	for i in range(num_epochs):
		train_loss = train_epoch(encoder,decoder,device, train_loader, loss, optim)
		train_loss_track.append(train_loss)
		print(f"training loss at {i} epoch = {train_loss}")
		if test:
			test_l = test_epoch(encoder,decoder,device, test_loader,loss)
			test_loss_track.append(test_l.item())
			print(f"test loss at {i} epoch = {test_l}")
	if save_model:
		torch.save(encoder,"encoder.pt")
		torch.save(decoder,"decoder.pt")
		torch.save(train_loss_track,"train_loss.pt")
		if test:
			torch.save(test_loss_track, "test_loss.pt")

	return (encoder,decoder, train_loss_track) if not test else (encoder,decoder,train_loss_track, test_loss_track)



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



