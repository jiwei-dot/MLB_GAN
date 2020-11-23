from torch.utils.data import DataLoader
import torch
import torchvision.datasets as dsets
import torchvision.transforms as T
import matplotlib.pyplot as plt


def get_real():
    dataset = dsets.MNIST(
        root='MNIST',
        train=True,
        transform=T.ToTensor(),
        download=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True
    )
    return dataloader


def get_noise(batchsize, dim, dtype=torch.float32, device='cpu'):
    noise = torch.rand((batchsize, dim), dtype=dtype, device=device)
    noise = (noise - 0.5) * 2
    return noise


if __name__ == '__main__':
    real_data = get_real()
    noise = get_noise(64, 96)
    print(noise.shape)
    for x, y in real_data:
        print(x.shape)
        plt.imshow(x[0].permute(1, 2, 0).squeeze(-1), cmap='gray')
        plt.title('digit = {}'.format(y[0].item()))
        plt.show()
        break
