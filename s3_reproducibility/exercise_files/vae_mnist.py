"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
from hydra import compose, initialize
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Model Hyperparameters
initialize(config_path=".", job_name="vae_mnist")
config = compose(config_name="config.yaml")

dataset_path = config.hyperparameters.dataset_path
batch_size = config.hyperparameters.batch_size
x_dim = config.hyperparameters.x_dim
hidden_dim = config.hyperparameters.hidden_dim
latent_dim = config.hyperparameters.latent_dim
lr = config.hyperparameters.lr
epochs = config.hyperparameters.epochs
cuda = config.hyperparameters.cuda
random_seed = config.hyperparameters.random_seed

# For reproducibility
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

dataset_path = "~/datasets"
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# Data loading
mnist_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(encoder=encoder, decoder=decoder).to(DEVICE)


def loss_function(x, x_hat, mean, log_var):
    """Elbo loss function."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


optimizer = Adam(model.parameters(), lr=lr)


print("Start training VAE...")
model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(batch_idx)
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*batch_size)}")
print("Finish!!")

# save weights
torch.save(model, f"{os.getcwd()}/trained_model.pt")

# Generate reconstructions
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        if batch_idx % 100 == 0:
            print(batch_idx)
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        x_hat, _, _ = model(x)
        break

save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

# Generate samples
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")
