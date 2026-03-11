import torch
import torch.nn as nn

from models.multi_view_core.encoder import InteractionLayer, SelfAttention


class AnomalyHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.feature == "latent":
            if args.loss_type == "ALL":
                self.self_attention = SelfAttention(args.num_hidden)
            input_dim = len(args.loss_type) * args.num_hidden
        else:
            self.interaction_layer = InteractionLayer(args.num_embedding, args.num_head)

            self.output_layer_t = nn.Sequential(
                nn.Linear(args.num_embedding * 2, args.num_hidden),
                nn.LayerNorm(args.num_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden),
            )
            self.output_layer_d = nn.Sequential(
                nn.Linear(args.num_embedding * 2, args.num_hidden),
                nn.LayerNorm(args.num_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden),
            )
            self.output_layer_f = nn.Sequential(
                nn.Linear(args.num_embedding * 2, args.num_hidden),
                nn.LayerNorm(args.num_hidden),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.num_hidden, args.num_hidden),
            )
            input_dim = len(args.loss_type) * args.num_hidden

        self.projection = nn.Sequential(
            nn.Linear(input_dim, args.num_hidden),
            nn.LayerNorm(args.num_hidden),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.num_hidden, args.projection_dim),
        )

    def _select_domains(self, zt, zd, zf):
        if self.args.loss_type == "ALL":
            emb = torch.cat([zt, zd, zf], dim=-1)
        else:
            emb_list = []
            if "T" in self.args.loss_type:
                emb_list.append(zt)
            if "D" in self.args.loss_type:
                emb_list.append(zd)
            if "F" in self.args.loss_type:
                emb_list.append(zf)
            emb = torch.cat(emb_list, dim=-1)
        return emb

    def forward(self, xt, dx, xf):
        xt, dx, xf = torch.nan_to_num(xt), torch.nan_to_num(dx), torch.nan_to_num(xf)

        if self.args.feature == "latent":
            zt, zd, zf = xt, dx, xf
            if self.args.loss_type == "ALL":
                stacked_emb = torch.stack([zt, zd, zf], dim=1)
                fused = self.self_attention(stacked_emb)[0] + stacked_emb
                zt, zd, zf = fused[:, 0, :], fused[:, 1, :], fused[:, 2, :]
        else:
            ht, hd, hf = xt, dx, xf
            if self.args.loss_type == "ALL":
                ht_i, hd_i, hf_i = self.interaction_layer(ht, hd, hf)
            else:
                ht_i, hd_i, hf_i = ht, hd, hf

            zt = self.output_layer_t(torch.cat([ht.mean(dim=1), ht_i.mean(dim=1)], dim=-1))
            zd = self.output_layer_d(torch.cat([hd.mean(dim=1), hd_i.mean(dim=1)], dim=-1))
            zf = self.output_layer_f(torch.cat([hf.mean(dim=1), hf_i.mean(dim=1)], dim=-1))

        emb = self._select_domains(zt, zd, zf)
        emb = emb.reshape(emb.shape[0], -1)
        proj = self.projection(emb)
        return nn.functional.normalize(proj, dim=-1)
