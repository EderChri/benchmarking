import matplotlib.pyplot as plt
from plots.base_plot import BasePlot

class PointPlot(BasePlot):
    def _prediction_anomaly_mask(self, art, pred_df):
        pred_col = pred_df.columns[0]
        values = pred_df[pred_col].astype(float)
        unique_vals = set(values.dropna().unique().tolist())

        if unique_vals.issubset({0.0, 1.0}):
            return values > 0.5

        quantile = (
            art.get("model_config", {})
            .get("params", {})
            .get("threshold_quantile", 0.99)
        )
        threshold = values.quantile(float(quantile))
        return values >= threshold

    def plot(self, run_ids, artifacts, output_path, **_):
        fig, ax = plt.subplots(figsize=self.params.get("figsize", [12, 5]))
        alpha = self.params.get("alpha", 0.8)
        ms = self.params.get("marker_size", 4)
        for run_id, art in zip(run_ids, artifacts):
            name = art.get("experiment", f"run_{run_id}")
            task = str(art.get("run_cfg", {}).get("task", "")).lower()
            if (td := art.get("test_data")) is not None:
                test_df = td.to_pd()
                ax.scatter(test_df.index, test_df.iloc[:, 0], label=f"{name} (ground truth series)", alpha=alpha, s=ms)

                if task in ["anomaly", "anomaly_detection"] and (pred := art.get("predictions")) is not None:
                    pred_df = pred.to_pd()
                    pred_idx = test_df.index.intersection(pred_df.index)
                    if len(pred_idx) > 0:
                        pred_mask = self._prediction_anomaly_mask(art, pred_df.loc[pred_idx])
                        pred_x = pred_idx[pred_mask.values]
                        if len(pred_x) > 0:
                            pred_y = test_df.loc[pred_x, test_df.columns[0]]
                            ax.scatter(
                                pred_x,
                                pred_y,
                                color="orange",
                                marker="x",
                                s=max(20, ms * 2),
                                label=f"{name} (predicted anomaly)",
                                zorder=6,
                            )

                if task in ["anomaly", "anomaly_detection"] and (labels := art.get("test_labels")) is not None:
                    label_df = labels.to_pd()
                    common_idx = test_df.index.intersection(label_df.index)
                    if len(common_idx) > 0:
                        y = test_df.loc[common_idx, test_df.columns[0]]
                        label_col = label_df.columns[0]
                        anomaly_mask = label_df.loc[common_idx, label_col].astype(float) > 0
                        anomaly_x = common_idx[anomaly_mask.values]
                        anomaly_y = y.loc[anomaly_x]
                        if len(anomaly_x) > 0:
                            ax.scatter(
                                anomaly_x,
                                anomaly_y,
                                color="red",
                                marker="o",
                                s=max(20, ms * 2),
                                label=f"{name} (ground-truth anomaly)",
                                zorder=5,
                            )
            if (pred := art.get("predictions")) is not None:
                df = pred.to_pd()
                ax.scatter(df.index, df.iloc[:, 0], label=f"{name} (prediction signal)", alpha=alpha, s=ms, marker="x")
        ax.legend(); ax.grid(True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
