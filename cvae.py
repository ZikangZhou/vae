import torch

from base_vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Any


class CVAE(BaseVAE):

    def __init__(self, channels_list: List, latent_dim: int, num_classes: int) -> None:
        super(CVAE, self).__init__()
        self.channels_list = list(channels_list)
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.embed_class = nn.Linear(num_classes, 28 * 28)

        modules = []
        channels_list[0] += 1
        for i in range(len(channels_list) - 1):
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=1),
                nn.BatchNorm2d(channels_list[i + 1]),
                nn.ReLU()))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(channels_list[-1] * (28 - (len(channels_list) - 1) * 2) ** 2, latent_dim)
        self.fc_var = nn.Linear(channels_list[-1] * (28 - (len(channels_list) - 1) * 2) ** 2, latent_dim)

        channels_list.reverse()
        channels_list[-1] -= 1
        modules = []
        self.decoder_inputs = nn.Linear(latent_dim + num_classes,
                                        channels_list[0] * (28 - (len(channels_list) - 1) * 2) ** 2)
        for i in range(len(channels_list) - 2):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3,
                                   stride=1),
                nn.BatchNorm2d(channels_list[i + 1]),
                nn.ReLU()))
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=channels_list[-2], out_channels=channels_list[-1], kernel_size=3, stride=1),
            nn.BatchNorm2d(channels_list[-1]),
            nn.Sigmoid()))
        self.decoder = nn.Sequential(*modules)

    def encode(self, inputs: torch.tensor) -> List[torch.tensor]:
        x = self.encoder(inputs)
        x = x.view(-1, self.channels_list[-1] * (28 - (len(self.channels_list) - 1) * 2) ** 2)
        return [self.fc_mu(x), self.fc_var(x)]

    def decode(self, inputs: torch.tensor) -> Any:
        x = self.decoder_inputs(inputs)
        x = x.view(-1, self.channels_list[-1], (28 - (len(self.channels_list) - 1) * 2),
                   (28 - (len(self.channels_list) - 1) * 2))
        return self.decoder(x)

    @staticmethod
    def reparameterize(mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, *inputs: torch.tensor) -> List[torch.tensor]:
        x = inputs[0]
        y = inputs[1]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y = torch.zeros(y.size(0), self.num_classes).to(y.device).scatter_(1, y, 1)
        embed_class = self.embed_class(y)
        embed_class = embed_class.view(-1, 1, 28, 28)
        x = torch.cat([x, embed_class], dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, y], dim=1)
        return [self.decode(z), mu, logvar]

    def loss_function(self, *inputs: Any, **kwargs) -> torch.tensor:
        recon_x = inputs[0]
        x = inputs[1]
        mu = inputs[2]
        logvar = inputs[3]
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def sample(self, batch_size: int, current_device: torch.device, **kwargs) -> torch.tensor:
        y = kwargs['labels']
        if y.dim() == 1:
            y = y.unsqueeze(1)
        y = torch.zeros(y.size(0), self.num_classes).scatter_(1, y, 1).to(current_device)
        z = torch.randn(batch_size, self.latent_dim)
        z = z.to(current_device)
        z = torch.cat([z, y], dim=1)
        return self.decode(z).cpu()
