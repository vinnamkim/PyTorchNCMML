import torch
from torch import nn


class NCMMLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                features: torch.Tensor,
                features_mean: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert features.dim() == 2 and features_mean.dim() == 2

        dists = (features.unsqueeze(1) -
                 features_mean.unsqueeze(0)).norm(dim=-1)
        return self.loss(-dists, targets)


class LinearTransform(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_method: str = 'random',
                 bias: bool = False):
        super().__init__()
        self.transform = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias)

        with torch.no_grad():
            if init_method == 'random':
                self.transform.weight.data = 0.1 * \
                    torch.randn_like(self.transform.weight.data)
            elif init_method == 'identity':
                self.transform.weight.data = torch.zeros_like(
                    self.transform.weight.data)
                self.transform.weight.data.fill_diagonal_(1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.transform(features)
