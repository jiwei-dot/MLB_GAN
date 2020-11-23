from data import get_noise
from model import Generator
from utils import show_images
import torch

z0 = get_noise(10, 96)
z1 = get_noise(10, 96)
w = torch.linspace(0, 1, 10).view(-1, 1, 1)
G = Generator(input_dim=96)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G.load_state_dict(torch.load('results/generator.pth', map_location=device))
G.eval()

# latent space interpolation
# use pytorch broadcasting
z = (1 - w) * z0 + w * z1    # w x B x D
z = z.view(-1, 96)
G_z = G(z).detach()
show_images(G_z)

