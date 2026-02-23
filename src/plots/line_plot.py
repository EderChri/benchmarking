import matplotlib.pyplot as plt
from plots.base_plot import BasePlot

class LinePlot(BasePlot):
    def plot(self, run_ids, artifacts, output_path):
        fig, ax = plt.subplots(figsize=self.params.get("figsize", [12, 5]))
        alpha = self.params.get("alpha", 0.8)
        for run_id, art in zip(run_ids, artifacts):
            name = art.get("experiment", f"run_{run_id}")
            if (td := art.get("test_data")) is not None:
                df = td.to_pd()
                ax.plot(df.index, df.iloc[:, 0], linestyle="-", label=f"{name} (ground truth)", alpha=alpha)
            if (pred := art.get("predictions")) is not None:
                df = pred.to_pd()
                ax.plot(df.index, df.iloc[:, 0], linestyle="--", label=f"{name} (prediction)", alpha=alpha)
        ax.legend(); ax.grid(True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
