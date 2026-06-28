import argparse

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt


class Autoencoder(nn.Module):

    def __init__(self, latent_dim=32):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,latent_dim)

        )

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim,128),
            nn.ReLU(),

            nn.Linear(128,256),
            nn.ReLU(),

            nn.Linear(256,784),
            nn.Sigmoid()

        )

    def forward(self,x):

        z=self.encoder(x)

        x=self.decoder(z)

        return x.view(-1,1,28,28)


parser=argparse.ArgumentParser()

parser.add_argument("--model",required=True)

parser.add_argument("--index",type=int,default=0)

args=parser.parse_args()

transform=transforms.ToTensor()

dataset=datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

image,_=dataset[args.index]

model=Autoencoder(32)

model.load_state_dict(torch.load(args.model,map_location="cpu"))

model.eval()

with torch.no_grad():

    output=model(image.unsqueeze(0))

original=image.squeeze().numpy()

reconstructed=output.squeeze().numpy()

plt.imsave("original.png",original,cmap="gray")

plt.imsave("reconstructed.png",reconstructed,cmap="gray")

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original,cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed,cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.savefig("comparison.png")

plt.show()

print("original.png berhasil dibuat")

print("reconstructed.png berhasil dibuat")

print("comparison.png berhasil dibuat")