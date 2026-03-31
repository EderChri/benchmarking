import torch
import torch.nn as nn


class RawForecastModel(nn.Module):
    """
    Sequence-to-forecast model that operates on raw input windows. Equivalent to the head of multi_view_forecaster.py

    Projects each time step to an embedding, mean-pools over the sequence,
    then refines through a projection MLP before forecasting.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_out = getattr(args, 'num_out_features', 1)

        self.feature_embedding = nn.Linear(args.num_feature, args.num_embedding)
        self.projection = nn.Sequential(
            nn.Linear(args.num_embedding, args.num_hidden),
            nn.LayerNorm(args.num_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.num_hidden),
        )
        self.forecast = nn.Linear(args.num_hidden, args.forecast_horizon * self.num_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [batch, seq, num_feature]
        Returns:
            [batch, horizon] if num_out=1, else [batch, horizon, num_out]
        """
        x = torch.nan_to_num(x)
        z = self.feature_embedding(x).mean(dim=1)  # [batch, num_embedding]
        z = self.projection(z)                      # [batch, num_hidden]
        out = self.forecast(z).reshape(x.shape[0], self.args.forecast_horizon, self.num_out)
        return out.squeeze(-1)
