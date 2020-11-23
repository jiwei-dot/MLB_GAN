import torch.nn as nn


# MLP
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# another MLP
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    import torch
    z = torch.rand(64, 96)
    G = Generator(input_dim=96)
    D = Discriminator()
    G_z = G(z)
    print(G_z.shape)
    D_G_z = D(G_z)
    print(D_G_z.shape)


