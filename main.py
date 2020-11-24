import torch

from runner import Runner
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vanilla_vae import VanillaVAE


batch_size = 128
epochs = 500


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaVAE([1, 8, 16], 16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    runner = Runner(model, optimizer, train_loader, test_loader, device)
    for epoch in range(1, epochs + 1):
        runner.train(epoch)
        runner.test(epoch)
        with torch.no_grad():
            sample = model.sample(64, device)
            save_image(sample, './results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    main()
