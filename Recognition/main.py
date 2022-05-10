from siameseNetworkDataset import SiameseNetworkDataset
from siameseNetwork import SiameseNetwork
from sklearn import metrics
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision.utils
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import splitfolders
import pandas as pd

img_folder = 'Dataset'
splitfolders.ratio(img_folder, output="Data", ratio=(.8, .1, .1), group_prefix=None)

folder_dataset = datasets.ImageFolder(root="./DifferentData/train/")

# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((105,105)),transforms.ToTensor()])

# Initialize the network
train_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,transform=transformation)

# Load the training dataset
train_dataloader = DataLoader(train_dataset,
						shuffle=True,
						num_workers=0,
						batch_size=8)

folder_dataset = datasets.ImageFolder(root="./DifferentData/val/")
val_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,transform=transformation)
# Load the training dataset
val_dataloader = DataLoader(val_dataset,
						shuffle=True,
						num_workers=0,
						batch_size=8)

net = SiameseNetwork().cuda()
#criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005 )
criterion = nn.BCEWithLogitsLoss()

# saving and loading checkpoint mechanisms
def save_checkpoint(save_path, model, optimizer, val_loss):
	if save_path==None:
		return
	save_path = save_path 
	state_dict = {'model_state_dict': model.state_dict(),
				  'optimizer_state_dict': optimizer.state_dict(),
				  'val_loss': val_loss}

	torch.save(state_dict, save_path)

	print(f'Model saved to ==> {save_path}')

# training and validation loss were calculated after every epoch
def train(model, train_loader, val_loader, num_epochs, criterion):
	best_val_loss = float("Inf") 
	train_losses = []
	val_losses = []
	cur_step = 0
	avg = 0
	pred = []
	for epoch in range(num_epochs):
		running_loss = 0.0
		running_acc = 0.0
		model.train()
		print("Starting epoch " + str(epoch+1))
		for img1, img2, labels in train_loader:
			
			# Forward
			img1 = img1.cuda()
			img2 = img2.cuda()
			labels = labels.cuda()
			outputs = model(img1, img2)
			loss = criterion(outputs, labels)
			# Calculating Train Acc
			output_array = outputs.cpu().detach().numpy()
			for i in range(len(output_array)):
				if output_array[i] > 0.5:
					pred.append(1)
				else:
					pred.append(0)	
			avg = metrics.accuracy_score(labels.cpu().detach().numpy(), pred)
			pred = []
			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			running_acc += avg
		avg_train_acc = running_acc / len(train_loader)
		avg_train_loss = running_loss / len(train_loader)
		train_losses.append(avg_train_loss)
		val_running_loss = 0.0
		val_running_acc = 0.0
		
		#check validation loss after every epoch
		with torch.no_grad():
			model.eval()
			for img1, img2, labels in val_loader:
				img1 = img1.cuda()
				img2 = img2.cuda()
				labels = labels.cuda()
				outputs = model(img1, img2)
				loss = criterion(outputs, labels)

				output_array = outputs.cpu().detach().numpy()
				for i in range(len(output_array)):
					if output_array[i] > 0.5:
						pred.append(1)
					else:
						pred.append(0)	
				avg = metrics.accuracy_score(labels.cpu().detach().numpy(), pred)
				pred = []
				val_running_acc += avg
				val_running_loss += loss.item()
		avg_val_loss = val_running_loss / len(val_loader)
		avg_val_acc = val_running_acc / len(val_dataloader)
		val_losses.append(avg_val_loss)
		print('Epoch [{}/{}],Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.8f}, Valid Acc: {:.8f}'
			.format(epoch+1, num_epochs, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))
		#if avg_val_loss < best_val_loss:
			#best_val_loss = avg_val_loss
			#save_checkpoint("Differentmodel.pth", model, optimizer, best_val_loss)
	
	print("Finished Training")  
	return train_losses, val_losses

train_losses, val_losses = train(net, train_dataloader, val_dataloader, 10, criterion)

#plotting of training and validation loss
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label="Validation Loss")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()