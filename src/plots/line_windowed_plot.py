import logging

import pandas as pd
from pandas import Timedelta
from merlion.utils import TimeSeries

from plots.line_plot import LinePlot

logger = logging.getLogger(__name__)

_DURATIONS = {
    "day": Timedelta(days=1),
    "week": Timedelta(weeks=1),
}


class LineWindowedPlot(LinePlot):
    def plot(self, run_ids, artifacts, output_path, start_time=None, **_):
        duration_key = self.params.get("duration", "week")
        duration = _DURATIONS.get(str(duration_key).lower())
        if duration is None:
            raise ValueError(f"Unknown duration '{duration_key}'. Use one of: {list(_DURATIONS)}")

        sliced = []
        for run_id, art in zip(run_ids, artifacts):
            td = art.get("test_data")
            if td is None:
                sliced.append(art)
                continue

            data_start = td.to_pd().index[0]
            data_end = td.to_pd().index[-1]

            if start_time is not None:
                try:
                    t0 = pd.Timestamp(start_time)
                except Exception:
                    logger.warning(f"[run {run_id}] Cannot parse start_time='{start_time}'; using data start.")
                    t0 = data_start
                if t0 < data_start or t0 >= data_end:
                    logger.warning(
                        f"[run {run_id}] start_time='{start_time}' is outside data range "
                        f"[{data_start}, {data_end}]. Using data start instead."
                    )
                    t0 = data_start
            else:
                t0 = data_start

            t1 = t0 + duration
            if t1 > data_end:
                logger.warning(
                    f"[run {run_id}] Window [{t0}, {t1}) exceeds data end ({data_end}). Plot will be truncated."
                )

            sliced.append({**art, "test_data": self._slice_ts(td, t0, t1), "predictions": self._slice_ts(art.get("predictions"), t0, t1)})

        super().plot(run_ids, sliced, output_path)

    def _slice_ts(self, ts, t0: pd.Timestamp, t1: pd.Timestamp):
        if ts is None:
            return None
        return ts.window(t0.timestamp(), t1.timestamp())
