import torch
import torch.nn as nn


class LinearForecastHead(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.num_out = getattr(args, 'num_out_features', 1)
		input_dim = len(args.loss_type) * args.num_hidden
		self.linear = nn.Linear(input_dim, args.forecast_horizon * self.num_out)

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
		else:
			ht, hd, hf = xt, dx, xf
			zt = ht.mean(dim=1)
			zd = hd.mean(dim=1)
			zf = hf.mean(dim=1)

		emb = self._select_domains(zt, zd, zf)
		emb = emb.reshape(emb.shape[0], -1)
		out = self.linear(emb).reshape(emb.shape[0], self.args.forecast_horizon, self.num_out)
		return out.squeeze(-1)  # [batch, horizon] if num_out=1, else [batch, horizon, num_out]
