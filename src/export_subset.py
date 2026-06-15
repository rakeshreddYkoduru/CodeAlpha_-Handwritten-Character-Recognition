import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

def export_subset(num_samples_per_class=5):
    """
    Exports a tiny subset of EMNIST as images for GitHub visualization.
    """
    output_dir = './dataset_preview'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    
    # Download EMNIST Balanced
    dataset = datasets.EMNIST(
        root='./data', 
        split='balanced', 
        train=True, 
        download=True, 
        transform=transform
    )

    mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    class_counts = {i: 0 for i in range(47)}
    
    print(f"Exporting {num_samples_per_class} samples per class to {output_dir}...")

    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        
        if class_counts[label] < num_samples_per_class:
            # Convert tensor to image
            # EMNIST images are transposed by default, so we fix them here
            img_np = img_tensor.numpy().squeeze().T * 255
            img = Image.fromarray(img_np.astype(np.uint8))
            
            char = mapping[label]
            char_dir = os.path.join(output_dir, f"class_{label}_{char}")
            if not os.path.exists(char_dir):
                os.makedirs(char_dir)
            
            img.save(os.path.join(char_dir, f"sample_{class_counts[label]}.png"))
            class_counts[label] += 1
            
        if all(count >= num_samples_per_class for count in class_counts.values()):
            break

    print("Export complete. You can now upload 'dataset_preview' to GitHub.")

if __name__ == "__main__":
    export_subset()
