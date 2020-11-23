from data import get_noise, get_real
from model import Generator, Discriminator
from utils import show_images
import torch
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(input_dim=96).to(device)
D = Discriminator().to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
G_criterion = torch.nn.BCEWithLogitsLoss()
D_criterion = torch.nn.BCEWithLogitsLoss()
EPOCHES = 10

real = get_real()
iter = 0
for epoch in range(EPOCHES):
    G.train()
    D.train()
    for x, _ in real:
        B = x.shape[0]
        x = x.view(B, -1)
        x = x.to(device)
        D_x = D((x - 0.5) * 2)
        d_real_loss = D_criterion(D_x, torch.ones_like(D_x))

        z = get_noise(128, 96, dtype=x.dtype, device=x.device)
        G_z = G(z).detach()
        D_G_z = D(G_z)
        d_fake_loss = D_criterion(D_G_z, torch.zeros_like(D_G_z))

        d_loss = d_real_loss + d_fake_loss
        D_optimizer.zero_grad()
        d_loss.backward()
        D_optimizer.step()

        z = get_noise(64, 96, dtype=x.dtype, device=x.device)
        G_z = G(z)
        D_G_z = D(G_z)
        # minimum log(1-D(G(z))) --> maximum log(D(G(z))) -->  minimum -log(D(G(z))) -->  negative log likelihood
        g_loss = G_criterion(D_G_z, torch.ones_like(D_G_z))
        G_optimizer.zero_grad()
        g_loss.backward()
        G_optimizer.step()
        if iter % 500 == 0:
            print('iter:{}\tg_loss:{:.3f}\td_loss:{:.3f}'.format(iter, g_loss.item(), d_loss.item()))
            # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            fake_images_cpu = G_z.data.cpu()
            show_images(fake_images_cpu[:16])
        iter += 1

torch.save(G.state_dict(), 'results/generator.pth')
torch.save(D.state_dict(), 'results/discriminator.pth')


