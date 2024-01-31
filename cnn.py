import torch
import torch.nn as nn
import torch.optim as optim
from torch import save
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from dataset import ShapeDataset
from shapecnn import ShapeCNN
from util import transform_training as transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
	"batch_size": 32,
	"learning_rate": 0.0002,
	"num_epochs": 100,
	"num_classes": 3,
}

if __name__ == '__main__':
    # Load dataset
    dataset = ShapeDataset("./dataset", transform=transform)

    # Divide dataset to train and test subsets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Create model, CrossEntropyLoss and Adam optimizer
    model = ShapeCNN(num_classes=CONFIG['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Train model
    for epoch in range(CONFIG['num_epochs']):
        for inputs, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Loss: {loss.item()}")

    # Evaluate model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test set: {accuracy}")

    # Save model
    save(model.state_dict(), 'model.pth')