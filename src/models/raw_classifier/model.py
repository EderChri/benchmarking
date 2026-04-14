import torch
import torch.nn as nn


class RawHead(nn.Module):
    """Mirrors the Classifier's feature="latent" path without an encoder.

    Each view is avg-pooled over time to [batch, num_feature], then the
    relevant views are concatenated and fed through a single linear layer.

    input_mode="raw"       : xt only  -> fc( num_feature,   num_target)
    input_mode="multi_view" : xt+dx+xf -> fc(3*num_feature,  num_target)
    """

    def __init__(self, num_feature: int, num_target: int, input_mode: str):
        super().__init__()
        self.input_mode = input_mode
        n_views = 3 if input_mode == "multi_view" else 1
        self.fc = nn.Linear(n_views * num_feature, num_target)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1) if x.dim() == 3 else x

    def forward(self, xt: torch.Tensor, dx: torch.Tensor = None,
                xf: torch.Tensor = None) -> torch.Tensor:
        xt = torch.nan_to_num(xt)
        if self.input_mode == "multi_view":
            emb = torch.cat([
                self._pool(xt),
                self._pool(torch.nan_to_num(dx)),
                self._pool(torch.nan_to_num(xf)),
            ], dim=-1)
        else:
            emb = self._pool(xt)
        return self.fc(emb)
