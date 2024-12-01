import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28*28
num_epochs = 100
batch_size = 64
lr = 0.01

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize networks and optimizers
generator = Generator()
discriminator = Discriminator()
# Changed from Adam to SGD
g_optimizer = optim.SGD(generator.parameters(), lr=lr)
d_optimizer = optim.SGD(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

def train_gan():
    # Keep track of generated images for each milestone
    generated_images = []
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.view(-1, image_dim)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size, latent_dim)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'd_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
        
        # Generate and save images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\nGenerating images at epoch {epoch+1}")
            with torch.no_grad():
                z = torch.randn(16, latent_dim)
                fake_images = generator(z)
                fake_images = fake_images.reshape(-1, 28, 28)
                generated_images.append((epoch + 1, fake_images))
                
            # Display the current state
            plt.figure(figsize=(10, 10))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(fake_images[i].detach(), cmap='gray')
                plt.axis('off')
            plt.suptitle(f'Epoch {epoch+1}')
            plt.show()

    return generator, generated_images

# Create an animation of the training progress
def create_training_progress_grid(generated_images):
    num_milestones = len(generated_images)
    plt.figure(figsize=(15, 2 * num_milestones))
    
    for idx, (epoch, images) in enumerate(generated_images):
        for i in range(8):  # Show first 8 images from each milestone
            plt.subplot(num_milestones, 8, idx * 8 + i + 1)
            plt.imshow(images[i].detach(), cmap='gray')
            if i == 0:
                plt.ylabel(f'Epoch {epoch}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Train the model and display progress
if __name__ == "__main__":
    trained_generator, progress_images = train_gan()
    print("\nCreating final training progress visualization...")
    create_training_progress_grid(progress_images)
