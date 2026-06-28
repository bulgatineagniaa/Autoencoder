import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Decoder(nn.Sequential):
    def __init__(self, latent_dim=32):
        super().__init__(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--decoder",
    type=str,
    required=True,
    help="Path decoder model (.pth)"
)

parser.add_argument(
    "--latent",
    type=str,
    required=True,
    help='Contoh: "0.1,0.2,0.3,... (32 angka)"'
)

args = parser.parse_args()

latent_values = [float(i) for i in args.latent.split(",")]

if len(latent_values) != 32:
    raise ValueError("Latent vector harus berisi 32 angka.")

decoder = Decoder(32)

decoder.load_state_dict(torch.load(args.decoder, map_location="cpu"))

decoder.eval()

z = torch.tensor(latent_values).float().unsqueeze(0)

with torch.no_grad():
    output = decoder(z)

image = output.view(28,28).numpy()

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.savefig("generated_image.png", bbox_inches="tight")
plt.show()

print("generated_image.png berhasil dibuat.")