import torch
import torch.nn as nn
import torch.nn.functional as F
#create the Siamese Neural Network
class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		# Conv2d(input_channels, output_channels, kernel_size)
		self.conv1 = nn.Conv2d(1, 64, 10) 
		self.conv2 = nn.Conv2d(64, 128, 7)  
		self.conv3 = nn.Conv2d(128, 128, 4)
		self.conv4 = nn.Conv2d(128, 256, 4)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(128)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.dropout1 = nn.Dropout(0.1)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(256 * 6 * 6, 4096)
		self.fcOut = nn.Linear(4096, 1)
		self.sigmoid = nn.Sigmoid()
	
	def convs(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.max_pool2d(x, (2,2))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.max_pool2d(x, (2,2))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.max_pool2d(x, (2,2))
		x = F.relu(self.bn4(self.conv4(x)))
		return x

	def forward(self, x1, x2):
		x1 = self.convs(x1)
		x1 = x1.view(-1, 256 * 6 * 6)
		x1 = self.sigmoid(self.fc1(x1))
		x2 = self.convs(x2)
		x2 = x2.view(-1, 256 * 6 * 6)
		x2 = self.sigmoid(self.fc1(x2))
		x = torch.abs(x1 - x2)
		x = self.fcOut(x)
		return x