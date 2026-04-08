import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from plots.base_plot import BasePlot

_DAY_ABBR = {0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"}


def _day_date_formatter(x, pos):
    dt = mdates.num2date(x)
    return f"{_DAY_ABBR[dt.weekday()]}\n{dt.strftime('%Y-%m-%d')}"

class LinePlot(BasePlot):
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

    def _forecast_feature_cols(self, run_ids, artifacts):
        """Return the list of feature column names to use for forecast subplots."""
        for run_id, art in zip(run_ids, artifacts):
            task = str(art.get("run_cfg", {}).get("task", "")).lower()
            if task not in ["forecast", "forecasting"]:
                continue
            pred = art.get("predictions")
            if pred is not None:
                cols = pred.to_pd().columns.tolist()
                if len(cols) > 1:
                    return cols
        return None  # univariate or no forecasting run found

    def plot(self, run_ids, artifacts, output_path, **kwargs):
        params = {**self.params, **kwargs}
        alpha = params.get("alpha", 0.8)
        feature_cols = self._forecast_feature_cols(run_ids, artifacts)
        n_subplots = len(feature_cols) if feature_cols else 1
        subplot_height = params.get("subplot_height", 4)
        figsize = params.get("figsize", [16, subplot_height * n_subplots])
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, squeeze=False)

        n_runs = len(run_ids)
        for run_id, art in zip(run_ids, artifacts):
            name = art.get("experiment", f"run_{run_id}")
            task = str(art.get("run_cfg", {}).get("task", "")).lower()
            pred = art.get("predictions")
            label_actual = "Actual" if n_runs == 1 else f"{name} – actual"
            label_pred = "Predicted" if n_runs == 1 else f"{name} – predicted"

            if task in ["forecast", "forecasting"] and (td := art.get("test_data")) is not None:
                test_df = td.to_pd()
                pred_df = pred.to_pd() if pred is not None else None

                if feature_cols is not None:
                    for i, col in enumerate(feature_cols):
                        ax = axes[i][0]
                        ax.set_title(col)
                        if col in test_df.columns:
                            ax.plot(test_df.index, test_df[col], linestyle="-", label=label_actual, alpha=alpha)
                        if pred_df is not None and col in pred_df.columns:
                            ax.plot(pred_df.index, pred_df[col], linestyle="--", label=label_pred, alpha=alpha)
                else:
                    ax = axes[0][0]
                    ax.set_title(test_df.columns[0])
                    ax.plot(test_df.index, test_df.iloc[:, 0], linestyle="-", label=label_actual, alpha=alpha)
                    if pred_df is not None:
                        ax.plot(pred_df.index, pred_df.iloc[:, 0], linestyle="--", label=label_pred, alpha=alpha)
                continue

            # Anomaly / classification — always on axes[0]
            ax = axes[0][0]
            if (td := art.get("test_data")) is not None:
                test_df = td.to_pd()
                ax.plot(test_df.index, test_df.iloc[:, 0], linestyle="-", label=f"{name} (ground truth)", alpha=alpha)

                if task in ["anomaly", "anomaly_detection"] and pred is not None:
                    pred_df = pred.to_pd()
                    pred_idx = test_df.index.intersection(pred_df.index)
                    if len(pred_idx) > 0:
                        pred_mask = self._prediction_anomaly_mask(art, pred_df.loc[pred_idx])
                        pred_x = pred_idx[pred_mask.values]
                        if len(pred_x) > 0:
                            pred_y = test_df.loc[pred_x, test_df.columns[0]]
                            ax.scatter(pred_x, pred_y, color="orange", marker="x", s=34,
                                       label=f"{name} (predicted anomaly)", zorder=6)

                if task in ["anomaly", "anomaly_detection"] and (labels := art.get("test_labels")) is not None:
                    label_df = labels.to_pd()
                    label_idx = test_df.index.intersection(label_df.index)
                    if len(label_idx) > 0:
                        y = test_df.loc[label_idx, test_df.columns[0]]
                        label_col = label_df.columns[0]
                        gt_mask = label_df.loc[label_idx, label_col].astype(float) > 0
                        gt_x = label_idx[gt_mask.values]
                        gt_y = y.loc[gt_x]
                        if len(gt_x) > 0:
                            ax.scatter(gt_x, gt_y, color="red", marker="o", s=24,
                                       label=f"{name} (ground-truth anomaly)", zorder=5)

            if pred is not None:
                pred_df = pred.to_pd()
                ax.plot(pred_df.index, pred_df.iloc[:, 0], linestyle="--", label=f"{name} (prediction signal)", alpha=alpha)

        # Single shared legend to the right of all subplots, labels deduped
        seen, handles, labels = set(), [], []
        for ax_row in axes:
            for h, l in zip(*ax_row[0].get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), framealpha=0.9)

        for ax_row in axes:
            ax = ax_row[0]
            ax.xaxis.set_major_formatter(FuncFormatter(_day_date_formatter))
            ax.grid(True)
        fig.subplots_adjust(right=0.99, hspace=0.4)
        fig.autofmt_xdate(rotation=0, ha="center")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
