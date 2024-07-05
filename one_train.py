import datetime
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from device import get_device
from networks import ConvNet


def main():
    mp.set_start_method('spawn', force=True)

    ctx = get_device()
    print(ctx)

    # Define the transformation to apply to the images
    transform = transforms.Compose([
        # transforms.RandomRotation(360),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Disable ssl so the dataset download works
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load the EMNIST dataset
    train_dataset = datasets.EMNIST(root='data', split='letters', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='data', split='letters', train=False, download=True, transform=transform)

    # Create data loaders for training and testing
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=1)

    # Initialize the neural network
    model = ConvNet()
    # model = SimpleNN()
    model.to(ctx)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 10
    start = datetime.datetime.now()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(ctx), labels.to(ctx)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    end = datetime.datetime.now()
    print(end - start)
    # Evaluating the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(ctx), labels.to(ctx)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), 'convnet_emnist.pth')


if __name__ == "__main__":
    main()
