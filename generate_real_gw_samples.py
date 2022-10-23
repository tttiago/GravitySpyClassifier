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
DATASET_PATH = "./datasets/Real_GWs/Real_GWs_O3_v1"
VMAX = 25.5

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
seg_int = 32
detectors = ["H1", "L1"]
time_windows = [0.5, 1.0, 2.0, 4.0]
window_pad = 1
tres = (min(time_windows)) / (140 * max(time_windows) + window_pad * 2)

if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    else:
        print("Dataset already exists.")
        sys.exit(0)

    for event in tqdm(events):
        gps = event_gps(event)
        segment = (np.ceil(gps) - seg_int / 2, np.ceil(gps) + seg_int / 2)
        for detector in detectors:
            try:
                data = TimeSeries.fetch_open_data(
                    detector, *segment, sample_rate=sample_rate, cache=True, verbose=False
                )
                hq = data.q_transform(
                    frange=(10, 2048),
                    outseg=(
                        gps - max(time_windows) / 2 - window_pad,
                        gps + max(time_windows) / 2 + window_pad,
                    ),
                    tres=tres,
                )
            except ValueError:
                print(
                    f"There was a problem with the {event.split('-')[0]}_{detector} signal. Skipping it..."
                )
                continue
            max_point = int(2 * len(hq) / 5) + np.argmax(
                np.max(hq[int(2 * len(hq) / 5) : int(3 * len(hq) / 5), :], axis=1)
            )
            shift = -max(time_windows) / 2 - window_pad + max_point * tres
            for time_window in time_windows:
                plot = hq.plot(
                    figsize=(17, 14),
                    dpi=20,
                    xlim=[gps + shift - time_window / 2, gps + shift + time_window / 2],
                    vmin=0,
                    vmax=VMAX,
                )
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
