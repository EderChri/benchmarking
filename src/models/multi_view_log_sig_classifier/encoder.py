"""N-view encoder for the log-sig classifier.

NViewEncoder is a strict generalisation of the existing three-view Encoder in
models/multi_view_core/encoder.py.  When configured with

    views=["xt", "dx", "xf"]  and  view_dims={"xt": F, "dx": F, "xf": F}

the computation graph is identical to the original Encoder: same input
projections, same TransformerEncoder stacks, same NViewInteractionLayer
(which reduces to the original InteractionLayer at N=3), same output MLPs.

Architecture per view:
    input [batch, seq_len, view_dim]
    → nn.Linear(view_dim, num_embedding)
    → PositionalEncoding
    → TransformerEncoder (num_layers layers)
    → hidden state h_v  [batch, seq_len, num_embedding]

After all branches:
    NViewInteractionLayer: stacks N hidden states, applies shared MHA across
    the N×seq_len tokens per batch item, unstacks back to N hidden states.

Per view output MLP:
    cat([h_v.mean(dim=1), h_v_interacted.mean(dim=1)])  →  [batch, num_embedding*2]
    → Linear(num_embedding*2, num_hidden) → LayerNorm → ReLU → Dropout
    → Linear(num_hidden, num_hidden)
    → latent z_v  [batch, num_hidden]

Returns:
    hiddens : dict[view_name → Tensor[batch, seq_len, num_embedding]]
    latents : dict[view_name → Tensor[batch, num_hidden]]
"""

import math
from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared building blocks (identical to multi_view_core/encoder.py)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class NViewInteractionLayer(nn.Module):
    """Shared self-attention applied independently to each view over the sequence dimension.

    Identical to the original InteractionLayer at N=3: each view's hidden states
    [batch, seq_len, dim] are refined by attending over their own timesteps via a
    shared MHA.  Views do not attend across each other — they are treated as
    independent batch items in the MHA call.

    Shape path:
        N × [batch, seq_len, dim]
        → stack → [batch, N, seq_len, dim]
        → reshape → [batch*N, seq_len, dim]   (each view is a separate "batch item")
        → MHA(h, h, h)                        (attend over seq_len)
        → LayerNorm(h + attn)
        → reshape → [batch, N, seq_len, dim]
        → unstack → N × [batch, seq_len, dim]
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            hidden_list: N tensors each [batch, seq_len, hidden_size]
        Returns:
            N tensors each [batch, seq_len, hidden_size]
        """
        N = len(hidden_list)
        batch, seq_len, dim = hidden_list[0].shape

        h = torch.stack(hidden_list, dim=1)               # [batch, N, seq_len, dim]
        h = h.contiguous().view(batch * N, seq_len, dim)  # [batch*N, seq_len, dim]

        attn_out, _ = self.multihead_attn(h, h, h)
        out = self.norm(h + attn_out)                     # [batch*N, seq_len, dim]
        out = out.view(batch, N, seq_len, dim)            # [batch, N, seq_len, dim]

        return [out[:, i, :, :] for i in range(N)]


def _make_output_mlp(num_embedding: int, num_hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(num_embedding * 2, num_hidden),
        nn.LayerNorm(num_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden, num_hidden),
    )


# ---------------------------------------------------------------------------
# NViewEncoder
# ---------------------------------------------------------------------------

class NViewEncoder(nn.Module):
    """N-view transformer encoder.

    Parameters
    ----------
    views : list[str]
        Ordered list of view names, e.g. ["xt", "dx", "xf"] or ["xt", "logsig"].
    view_dims : dict[str, int]
        Input feature dimension for each view.  For standard views (xt, dx, xf)
        this is num_feature.  For the logsig view this is logsig_dim (inferred
        from the data at model init time).
    num_embedding : int
        Transformer embedding dimension.
    num_hidden : int
        Output MLP hidden / output dimension.
    num_head : int
        Number of attention heads (must divide num_embedding).
    num_layers : int
        Number of TransformerEncoderLayer stacks per view.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        views: List[str],
        view_dims: Dict[str, int],
        num_embedding: int,
        num_hidden: int,
        num_head: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if not views:
            raise ValueError("views must contain at least one view name.")
        self.views = views

        self.positional_encoding = PositionalEncoding(num_embedding, dropout)

        # Per-view transformer branches stored in ModuleDicts.
        self.input_layers = nn.ModuleDict()
        self.transformers = nn.ModuleDict()
        self.output_mlps = nn.ModuleDict()

        for v in views:
            if v not in view_dims:
                raise ValueError(f"view_dims missing entry for view '{v}'.")
            in_dim = view_dims[v]
            self.input_layers[v] = nn.Linear(in_dim, num_embedding)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=num_embedding,
                dim_feedforward=num_hidden,
                nhead=num_head,
                dropout=dropout,
                batch_first=True,
            )
            self.transformers[v] = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_mlps[v] = _make_output_mlp(num_embedding, num_hidden, dropout)

        self.interaction_layer = NViewInteractionLayer(num_embedding, num_head)

    def forward(
        self, view_tensors: Dict[str, torch.Tensor]
    ) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        """
        Args:
            view_tensors: dict mapping each view name → Tensor[batch, seq_len, view_dim]

        Returns:
            hiddens: dict view_name → Tensor[batch, seq_len, num_embedding]
            latents: dict view_name → Tensor[batch, num_hidden]
        """
        # Sanitise inputs.
        for v in self.views:
            view_tensors[v] = torch.nan_to_num(view_tensors[v])

        # 1. Per-view input projection + positional encoding + transformer.
        hiddens_raw: Dict[str, torch.Tensor] = {}
        for v in self.views:
            h = self.input_layers[v](view_tensors[v])
            h = self.positional_encoding(h)
            h = self.transformers[v](h)
            hiddens_raw[v] = h   # [batch, seq_len, num_embedding]

        # 2. Cross-view interaction.
        hidden_list = [hiddens_raw[v] for v in self.views]
        interacted_list = self.interaction_layer(hidden_list)
        hiddens_interacted = {v: interacted_list[i] for i, v in enumerate(self.views)}

        # 3. Per-view output MLP: pool raw + interacted, project to num_hidden.
        hiddens: Dict[str, torch.Tensor] = {}
        latents: Dict[str, torch.Tensor] = {}
        for v in self.views:
            h_raw = hiddens_raw[v]
            h_int = hiddens_interacted[v]
            pooled = torch.cat([h_raw.mean(dim=1), h_int.mean(dim=1)], dim=-1)
            latents[v] = self.output_mlps[v](pooled)   # [batch, num_hidden]
            hiddens[v] = h_raw

        return hiddens, latents
