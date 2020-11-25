import torch
from torch import nn
from abc import abstractmethod
from typing import List, Any


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, inputs: torch.tensor) -> List[torch.tensor]:
        raise NotImplementedError

    def decode(self, inputs: torch.tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: torch.device, **kwargs) -> torch.tensor:
        raise RuntimeWarning()

    @abstractmethod
    def forward(self, *inputs: torch.tensor) -> List[torch.tensor]:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.tensor:
        pass
