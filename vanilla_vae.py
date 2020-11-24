import torch
from base_vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple


class VanillaVAE(BaseVAE):

    def __init__(self, channels_list: List, latent_dim: int) -> None:
        super(VanillaVAE, self).__init__()
        self.channels_list = list(channels_list)
        self.latent_dim = latent_dim

        modules = []
        for i in range(len(channels_list) - 1):
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=1),
                nn.BatchNorm2d(channels_list[i + 1]),
                nn.ReLU()))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(channels_list[-1] * (28 - (len(channels_list) - 1) * 2) ** 2, latent_dim)
        self.fc_var = nn.Linear(channels_list[-1] * (28 - (len(channels_list) - 1) * 2) ** 2, latent_dim)

        channels_list.reverse()
        modules = []
        self.decoder_inputs = nn.Linear(latent_dim,
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

    def forward(self, inputs: torch.tensor) -> List[torch.tensor]:
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
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
        z = torch.randn(batch_size, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z).cpu()

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return self.forward(x)[0]
