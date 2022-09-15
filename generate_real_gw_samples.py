from gwosc import datasets
from gwosc.datasets import event_gps, run_segment
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

GRAY_SCALE = True
DATASET_PATH = './datasets/Real_GWs_BW_v2'
VMAX = 25.5

O1_events = datasets.find_datasets(
    type="events", catalog="GWTC-1-confident", segment=run_segment("O1")
)
O2_events = datasets.find_datasets(
    type="events", catalog="GWTC-1-confident", segment=run_segment("O2_4KHZ_R1")
)
events = O1_events + O2_events

sample_rate = 16384
seg_int = 32
detectors = ["H1", "L1"]
time_windows = [0.5, 1.0, 2.0, 4.0]

for event in tqdm(events):
    gps = event_gps(event)
    segment = (np.ceil(gps) - seg_int/2, np.ceil(gps) + seg_int/2)
    for detector in detectors:
        data = TimeSeries.fetch_open_data(detector, *segment, sample_rate=sample_rate, cache=True, verbose=True)
        for time_window in time_windows:
            hq = data.q_transform(
                frange=(10, 2048), outseg=(gps - time_window / 2, gps + time_window / 2)
            )
            plot = hq.plot(figsize=(17, 14), dpi=20, vmin=0, vmax=VMAX)
            plt.yscale('log', base=2)
            ax = plt.gca()
            ax.axis("off")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.001)
            img_name = f"{DATASET_PATH}/{event.split('-')[0]}_{detector}_{time_window}.png"
            plt.savefig(img_name, dpi=10)
            plt.close()
            if GRAY_SCALE:
                Image.open(img_name).convert('L').save(img_name)