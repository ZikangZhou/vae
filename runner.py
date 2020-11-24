import torch

from base import BaseVAE
from torch import optim
from torchvision.utils import save_image


class Runner:

    def __init__(self, model: BaseVAE,
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
            recon_batch, mu, logvar = self.model(data)
            loss = self.model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                        100. * batch_idx / len(self.train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.model.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch.view(len(data), 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               './results/reconstruction_' + str(epoch) + '.png', nrow=n)
        test_loss /= len(self.test_loader.dataset)
        print('====> Epoch: {} Test set loss: {:.4f}'.format(epoch, test_loss))
