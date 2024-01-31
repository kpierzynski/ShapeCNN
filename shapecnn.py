import torch.nn as nn
from torch import flatten
from torch.nn.functional import softmax
from torch import no_grad, max

class ShapeCNN(nn.Module):
	def __init__(self, num_classes=3):
		super(ShapeCNN, self).__init__()

		self.conv1 = nn.Conv2d(
			in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
		)
		self.conv2 = nn.Conv2d(
			in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
		)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.fc1 = nn.Linear(32 * 64 * 64, 64)
		self.fc2 = nn.Linear(64, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool(x)

		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = softmax(x, dim=1)

		return x

	def predict(self, image, transform=None, device='cuda'):
		if transform:
			image = transform(image)

		image = image.unsqueeze(0)
		image = image.to(device)

		with no_grad():
			self.eval()
			output = self(image)

			confidence, prediction = max(output,1)
			index = prediction.item()
			return index, confidence.item()
