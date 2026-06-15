import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CharacterCNN
import os

def train(epochs=10, batch_size=128, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
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

    # Use 50% of dataset for feasible CPU training while maintaining quality
    subset_size = len(train_dataset) // 2
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_classes = 47 
    model = CharacterCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"Starting training for {num_classes} classes...")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_batches += 1
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/total_batches:.4f}")

        scheduler.step()
        
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

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss/total_batches:.4f}, Accuracy: {acc:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), './models/character_cnn.pth')
            print(f"  -> New best model saved! (Accuracy: {acc:.2f}%)")

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train(epochs=10)
