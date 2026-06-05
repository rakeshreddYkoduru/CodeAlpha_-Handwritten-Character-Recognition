import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CharacterCNN
import os

def train(epochs=15, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.RandomRotation(15), # Robustness to drawing angle
        transforms.RandomAffine(0, translate=(0.1, 0.1)), # Robustness to position
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,)) 
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1751,), (0.3332,))
    ])
   
    train_dataset = datasets.EMNIST(
        root='./data', 
        split='balanced', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.EMNIST(
        root='./data', 
        split='balanced', 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Reverted to 10% subset as requested
    subset_indices = torch.randperm(len(train_dataset))[:int(0.1 * len(train_dataset))]
    train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    test_indices = torch.randperm(len(test_dataset))[:int(0.1 * len(test_dataset))]
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 47 
    model = CharacterCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_classes} classes...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if i >= 100: # Keeping the 100 steps limit per epoch
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/100], Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1} completed. Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), './models/character_cnn.pth')
    print("Model saved to ./models/character_cnn.pth")

if __name__ == "__main__":
    train(epochs=5) # Start with 5 epochs for demonstration
