import torch
from torch import nn
from typing import Tuple


class NCMML(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mean_features: torch.Tensor,
                 transform: str = 'linear', init_method: str = 'random'):
        super().__init__()

        if transform == 'linear':
            self.transform = LinearTransform(
                in_features=in_features, out_features=out_features, init_method=init_method)
        else:
            raise Exception(f'Unknown transform : {transform}')

        self.mean_features = torch.tensor(mean_features)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, mean_features = self.transform(
            features), self.transform(self.mean_features.to(device=features.device))
        negative_dists = -(features.unsqueeze(1) -
                           mean_features.unsqueeze(0)).norm(dim=-1)

        outputs = (features, mean_features, negative_dists,)

        if targets is not None:
            loss = self.loss(negative_dists, targets)
            outputs = (loss,) + outputs

        return outputs

    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, logits = self.forward(features)
        _, indices = logits.topk(k=1, dim=-1)

        return logits, indices


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
