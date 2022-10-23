import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from gwosc import datasets
from gwosc.datasets import event_gps, run_segment
from gwpy.timeseries import TimeSeries
from PIL import Image
from tqdm import tqdm

GRAY_SCALE = True
DATASET_PATH = "./datasets/Real_GWs/Real_GWs_v10"

MAX_NORM_ENERGY = 25.5
Q_RANGE = (4, 64)
F_RANGE = (10, 2048)
T_RES = 0.002
F_RES = 0.5

O1_events = datasets.find_datasets(
    type="events", catalog="GWTC-1-confident", segment=run_segment("O1")
)
O2_events = datasets.find_datasets(
    type="events", catalog="GWTC-1-confident", segment=run_segment("O2_4KHZ_R1")
)
O3a_events = datasets.find_datasets(
    type="events", catalog="GWTC-2.1-confident", segment=run_segment("O3a_4KHZ_R1")
)
O3b_events = datasets.find_datasets(
    type="events", catalog="GWTC-3-confident", segment=run_segment("O3b_4KHZ_R1")
)
# events = O1_events + O2_events
events = O3a_events + O3b_events

sample_rate = 16384
detectors = ["H1", "L1"]
time_windows = [0.5, 1.0, 2.0, 4.0]

if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    else:
        print("Dataset already exists.")
        sys.exit(0)

    seg_int = 4 * np.max(time_windows)

    for event in tqdm(events):
        gps = event_gps(event)
        segment = (gps - seg_int / 2, gps + seg_int / 2)

        for detector in detectors:
            try:
                data = TimeSeries.fetch_open_data(
                    detector, *segment, sample_rate=sample_rate, cache=True, verbose=False
                )
                if np.isnan(data).any():
                    raise ValueError
            except ValueError:
                print(
                    f"There was a problem with the {event.split('-')[0]}_{detector} signal. Skipping it..."
                )
                continue

            for time_window in time_windows:
                try:
                    hq = data.q_transform(
                        qrange=Q_RANGE,
                        frange=F_RANGE,
                        gps=gps,
                        search=0.5,
                        tres=T_RES,
                        fres=F_RES,
                        outseg=(gps - time_window, gps + time_window),
                    )
                except Exception as e:
                    print(e)
                    hq = data.q_transform(
                        qrange=Q_RANGE,
                        frange=F_RANGE,
                        gps=gps,
                        search=0.5,
                        tres=T_RES,
                        fres=F_RES,
                        outseg=(gps - 2 * time_window, gps + 2 * time_window),
                    )

                hq = hq.crop(gps - time_window / 2, gps + time_window / 2)

                plot = hq.plot(figsize=(17, 14), dpi=20, vmin=0, vmax=MAX_NORM_ENERGY)
                plt.yscale("log", base=2)
                ax = plt.gca()
                ax.axis("off")
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.001)
                img_name = f"{DATASET_PATH}/{event.split('-')[0]}_{detector}_{time_window}.png"
                plt.savefig(img_name, dpi=10)
                plt.close()
                if GRAY_SCALE:
                    Image.open(img_name).convert("L").save(img_name)
