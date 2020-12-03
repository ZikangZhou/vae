import torch

from autoencoder import AutoEncoder
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

batch_size = 128
epochs = 200
channels_list = [1, 8, 16]
latent_dim = 16
lr = 1e-4


class AERunner:

    def __init__(self, model: AutoEncoder,
                 optimizer: optim.Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, z = self.model(data)
            loss = self.model.loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, z = self.model(data)
                test_loss += self.model.loss_function(recon_batch, data).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(len(data), 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               './results_ae_' + str(latent_dim) + '/reconstruction_' + str(epoch) + '.png', nrow=n)
        test_loss /= len(self.test_loader.dataset)
        print('====> Epoch: {} Test set loss: {:.4f}'.format(epoch, test_loss))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(channels_list, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    runner = AERunner(model, optimizer, train_loader, test_loader, device)
    for epoch in range(1, epochs + 1):
        runner.train(epoch)
        runner.test(epoch)
        with torch.no_grad():
            sample = model.sample(80, device)
            save_image(sample, './results_ae_' + str(latent_dim) + '/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    main()
