from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from .encoder import Encoder
from utils.config import marker


class MultiViewCoreMixin:
    def _get_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _create_encoder(self):
        args = Namespace(
            num_feature=self.config.num_feature,
            num_embedding=self.config.num_embedding,
            num_hidden=self.config.num_hidden,
            num_head=self.config.num_head,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        return Encoder(args)

    def _load_pretrained_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("encoder_state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)

    def _extract_domains_sequence(self, time_series: pd.DataFrame):
        if type(time_series) != pd.DataFrame:
            time_series = time_series.to_pd()

        num_feature = max(1, int(getattr(self.config, "num_feature", 1)))

        def _reshape_domain(values_2d: np.ndarray) -> np.ndarray:
            if values_2d.ndim != 2:
                return values_2d
            n, cols = values_2d.shape
            if cols % num_feature == 0:
                return values_2d.reshape(n, cols // num_feature, num_feature)
            return values_2d[:, :, np.newaxis]

        original_cols = [
            c for c in time_series.columns if marker not in str(c)
        ]
        derivative_cols = [c for c in time_series.columns if marker in str(c) and str(c).endswith("derivative")]
        fft_cols = [c for c in time_series.columns if marker in str(c) and str(c).endswith("fft")]

        xt = (
            time_series[original_cols].values
            if original_cols
            else np.zeros((len(time_series), 1), dtype=np.float32)
        )
        dx = time_series[derivative_cols].values if derivative_cols else np.zeros_like(xt)
        xf = time_series[fft_cols].values if fft_cols else np.zeros_like(xt)

        return _reshape_domain(xt), _reshape_domain(dx), _reshape_domain(xf)

    def _extract_domains_matrix(self, time_series: pd.DataFrame):
        num_feature = max(1, int(getattr(self.config, "num_feature", 1)))

        original_cols = [
            c for c in time_series.columns if marker not in str(c)
        ]
        derivative_cols = [c for c in time_series.columns if marker in str(c) and str(c).endswith("derivative")]
        fft_cols = [c for c in time_series.columns if marker in str(c) and str(c).endswith("fft")]

        xt = (
            time_series[original_cols].values.astype(np.float32)
            if original_cols
            else np.zeros((len(time_series), num_feature), dtype=np.float32)
        )
        dx = (
            time_series[derivative_cols].values.astype(np.float32)
            if derivative_cols
            else np.zeros_like(xt)
        )
        xf = (
            time_series[fft_cols].values.astype(np.float32)
            if fft_cols
            else np.zeros_like(xt)
        )

        if xt.shape[1] != num_feature:
            raise ValueError(
                f"Expected {num_feature} primary features, got {xt.shape[1]}. "
                "Set model.params.num_feature to match the number of non-domain columns."
            )
        if dx.shape[1] != num_feature:
            dx = np.zeros_like(xt)
        if xf.shape[1] != num_feature:
            xf = np.zeros_like(xt)

        target_col = int(min(max(self.config.target_seq_index, 0), xt.shape[1] - 1))
        return xt, dx, xf, target_col

    def _augment_data(self, xt, dx, xf):
        sigma = self.config.augmentation_strength
        perturb_ratio = 0.05

        xt_aug = xt + torch.normal(mean=0.0, std=sigma, size=xt.shape, device=xt.device)
        dx_aug = dx + torch.normal(mean=0.0, std=sigma, size=dx.shape, device=dx.device)

        mask_remove = (torch.rand(xf.shape, device=xf.device) > perturb_ratio).to(xf.dtype)
        xf_removed = xf * mask_remove
        mask_add = (torch.rand(xf.shape, device=xf.device) > (1 - perturb_ratio)).to(xf.dtype)
        max_amplitude = xf.max()
        random_am = torch.rand(xf.shape, device=xf.device) * (max_amplitude * 0.1)
        xf_added = xf + (mask_add * random_am)
        xf_aug = xf_removed + xf_added

        return xt_aug, dx_aug, xf_aug

    def _compute_loss_by_type(self, loss_t, loss_d, loss_f):
        loss_dict = {
            "ALL": loss_t + loss_d + loss_f,
            "TDF": loss_t + loss_d + loss_f,
            "TD": loss_t + loss_d,
            "TF": loss_t + loss_f,
            "DF": loss_d + loss_f,
            "T": loss_t,
            "D": loss_d,
            "F": loss_f,
        }
        return loss_dict.get(self.config.loss_type, loss_t + loss_d + loss_f)

    def _weight_regularization(self, model):
        l1_reg, l2_reg = 0.0, 0.0
        for param in model.parameters():
            if param.requires_grad:
                l1_reg += self.config.l1_scale * param.abs().sum()
                l2_reg += self.config.l2_scale * param.pow(2).sum()
        return l1_reg + l2_reg

    def _set_mode_for_encoder_and_head(self, head_module):
        mode = getattr(self.config, "mode", "finetune")

        if mode == "pretrain":
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True

        elif mode == "finetune":
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True
            head_module.train()

        elif mode == "freeze":
            self.encoder.eval()
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "input_layer" in name
            head_module.train()

    def _compute_contrastive_encoder_loss(
        self,
        batch_xt,
        batch_dx,
        batch_xf,
        batch_xt_aug,
        batch_dx_aug,
        batch_xf_aug,
    ):
        ht, hd, hf, zt, zd, zf = self.encoder(batch_xt, batch_dx, batch_xf)
        _, _, _, zt_aug, zd_aug, zf_aug = self.encoder(batch_xt_aug, batch_dx_aug, batch_xf_aug)

        loss_t = self.info_criterion(zt, zt_aug)
        loss_d = self.info_criterion(zd, zd_aug)
        loss_f = self.info_criterion(zf, zf_aug)

        contrastive_loss = self._compute_loss_by_type(loss_t, loss_d, loss_f) + self._weight_regularization(
            self.encoder
        )
        return (ht, hd, hf, zt, zd, zf), contrastive_loss

    def _compose_finetune_loss(self, contrastive_loss, supervised_loss, head_module):
        return self.config.lam * contrastive_loss + supervised_loss + self._weight_regularization(head_module)
