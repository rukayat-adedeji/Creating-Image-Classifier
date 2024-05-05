import torch
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import json
import argparse

# Function to load the dataset and create dataloaders
def load_data(data_dir):
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = ImageFolder(data_dir + '/train', transform=train_transforms)
    val_data = ImageFolder(data_dir + '/valid', transform=test_val_transforms)
    test_data = ImageFolder(data_dir + '/test', transform=test_val_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = DataLoader(val_data, batch_size=64)
    testloader = DataLoader(test_data, batch_size=64)

    return trainloader, valloader, testloader, train_data

# Load class-to-name mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
        
# Funtion to train the model
def train(data_dir, save_dir='checkpoint.pth', arch='densenet121', learning_rate=0.003, hidden_units=457, epochs=5, gpu=True):

    # Use GPU if specified
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    # Load pre-trained model
    model = models.__dict__[arch](pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Linear(1024, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    # load train_data variable
    train_data = load_data(data_dir)[3]
    model.class_to_idx = train_data.class_to_idx
    model.cat_to_name = cat_to_name

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Load data
    trainloader, valloader, _, _ = load_data(data_dir)    
    # Move model to device
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        print("Training....")
        model.train()
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        
        # Validation step
        model.eval()
        val_loss = 0
        accuracy = 0
        print("Validating....")
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                batch_loss = criterion(logps, labels)

                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {val_loss/len(valloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valloader):.3f}")
        

    # Save the model checkpoint
    checkpoint = {
        'input_size': 1024,
        'output_size': 102,
        'hidden_units': 457,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion,
        'class_to_idx': model.class_to_idx,
        'cat_to_name': model.cat_to_name,
        'arch': arch  # Save the chosen architecture in the checkpoint
    }

    torch.save(checkpoint, save_dir)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model checkpoint.')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save the model checkpoint')
    parser.add_argument('--arch', type=str, default='densenet121', help='Model architecture (e.g., densenet121)')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=457, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Train the model
    train(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
