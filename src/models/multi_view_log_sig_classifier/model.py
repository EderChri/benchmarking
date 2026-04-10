"""NViewClassifier head.

Receives the hiddens and latents dicts from NViewEncoder and produces class logits.

feature="latent"  (default, recommended)
    Takes the latent dict {view → [batch, num_hidden]}.
    Applies SelfAttention across the stacked N latent vectors, then concatenates
    and projects to num_target via a linear layer.
    Equivalent to the existing Classifier in latent+ALL mode at N=3.

feature="hidden"
    Takes the hidden dict {view → [batch, seq_len, num_embedding]}.
    Applies NViewInteractionLayer, pools each interacted hidden state, projects
    each to num_hidden via a per-view output MLP, then concatenates and classifies.
    Equivalent to the existing Classifier in hidden+ALL mode at N=3.

SelfAttention is unchanged from the original model.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from models.multi_view_log_sig_classifier.encoder import NViewInteractionLayer, _make_output_mlp


class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over N latent vectors (identical to original)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: [batch, N, hidden_dim]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights


class NViewClassifier(nn.Module):
    """Classification head for N-view encoder output.

    Parameters
    ----------
    views : list[str]
        Ordered list of view names — must match what NViewEncoder was built with.
    num_hidden : int
        Latent dimension per view (output of NViewEncoder).
    num_embedding : int
        Transformer hidden dimension (only needed for feature="hidden").
    num_head : int
        Attention heads (only needed for feature="hidden").
    num_target : int
        Number of output classes.
    dropout : float
        Dropout probability (only needed for feature="hidden").
    feature : str
        "latent" or "hidden".
    """

    def __init__(
        self,
        views: List[str],
        num_hidden: int,
        num_embedding: int,
        num_head: int,
        num_target: int,
        dropout: float,
        feature: str = "latent",
    ):
        super().__init__()
        self.views = views
        self.feature = feature
        N = len(views)

        if feature == "latent":
            self.self_attention = SelfAttention(num_hidden)
            fc_in = N * num_hidden

        elif feature == "hidden":
            self.interaction_layer = NViewInteractionLayer(num_embedding, num_head)
            self.output_mlps = nn.ModuleDict({
                v: _make_output_mlp(num_embedding, num_hidden, dropout)
                for v in views
            })
            fc_in = N * num_hidden

        else:
            raise ValueError(f"Unknown feature mode '{feature}'. Use 'latent' or 'hidden'.")

        self.fc = nn.Linear(fc_in, num_target)
        self.fc.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        hiddens: Dict[str, torch.Tensor],
        latents: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hiddens: view → [batch, seq_len, num_embedding]  (used in hidden mode)
            latents: view → [batch, num_hidden]              (used in latent mode)
        Returns:
            logits: [batch, num_target]
        """
        if self.feature == "latent":
            # Stack latents: [batch, N, num_hidden] → self-attention → concat → fc.
            stacked = torch.stack([latents[v] for v in self.views], dim=1)
            attended, _ = self.self_attention(stacked)   # [batch, N, num_hidden]
            # Residual connection (identical to original Classifier).
            attended = attended + stacked
            emb = attended.reshape(attended.size(0), -1)  # [batch, N*num_hidden]

        else:  # hidden
            hidden_list = [hiddens[v] for v in self.views]
            interacted = self.interaction_layer(hidden_list)
            # Pool each view and project.
            pooled = []
            for i, v in enumerate(self.views):
                h_raw = hiddens[v]
                h_int = interacted[i]
                p = torch.cat([h_raw.mean(dim=1), h_int.mean(dim=1)], dim=-1)
                pooled.append(self.output_mlps[v](p))
            emb = torch.cat(pooled, dim=-1)              # [batch, N*num_hidden]

        return self.fc(emb)
