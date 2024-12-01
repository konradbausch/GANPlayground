from sys import argv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import results
torch.manual_seed(42) #makes the randomnes seeded, so its reproducable

# params
latent_dim = 200
hidden_dim = 512 
image_dim = 28*28
num_epochs = 100
batch_size = 1024 * 2 * 2
lr = 0.0002

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# load MNIST 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=False)

def train_gan():
    result_images = []
    plot_data = []

    d_loss = 0
    g_loss = 0

    for epoch in range(num_epochs):
        for _, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.view(-1, image_dim)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train discriminator once
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

            # Train generator twice
            for _ in range(2):
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, latent_dim)  # New noise for each training step
                fake_images = generator(z)
                outputs = discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                g_optimizer.step()

        print(f"Training [Epoch: {epoch}]")

        # for later use
        if (epoch + 1) % 10 == 0:
            plot_data.append(f"{epoch+1}; {d_loss}; {g_loss}")
            with torch.no_grad():
                z = torch.randn(16, latent_dim)
                fake_images = generator(z)
                fake_images = fake_images.reshape(-1, 28, 28)
                result_images.append((epoch + 1, fake_images))

    return generator, result_images, plot_data

def create_training_progress_grid(generated_images, plot_data):
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

    output_path = argv[1]
    os.makedirs(output_path)
    plt.savefig(output_path + "/pics.png", dpi = 300)
    plt.clf()
    results.generate_plot(plot_data, output_path) 


if __name__ == "__main__":
    torch.set_num_threads(12)
    trained_generator, progress_images, plot_data = train_gan()
    create_training_progress_grid(progress_images, plot_data)

