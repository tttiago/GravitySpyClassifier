from gwosc import datasets
from gwosc.datasets import event_gps, run_segment
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import sys

GRAY_SCALE = True
SHIFT_SAMPLES = True
DATASET_PATH = './datasets/Real_GWs_v6'
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
window_pad = 1
tres = (max(time_windows)+2*window_pad)/4096

if __name__ == '__main__':
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    else:
        print('Dataset already exists.')
        sys.exit(0)
  
    for event in tqdm(events):
        gps = event_gps(event)
        segment = (np.ceil(gps) - seg_int/2, np.ceil(gps) + seg_int/2)
        for detector in detectors:
            data = TimeSeries.fetch_open_data(detector, *segment, sample_rate=sample_rate, cache=True, verbose=False)
            hq = data.q_transform(
                frange=(10, 2048), 
                outseg=(gps-max(time_windows)/2-window_pad, gps+max(time_windows)/2+window_pad),
                tres=tres,
                logf=True,
                whiten=True        
            )
            if SHIFT_SAMPLES:
                max_point = int(2*len(hq)/5) + np.argmax(np.max(hq[int(2*len(hq)/5):int(3*len(hq)/5),:], axis=1))
            else:
                max_point = 0
            shift = -max(time_windows)/2-window_pad + max_point * tres
            qspecgram = np.rot90(hq.value)
            
            for time_window in time_windows:            
                plt.axis('off')
                plt.imshow(qspecgram, vmin=0, vmax=VMAX)
                plt.xlim([max_point-int(len(hq)*time_window/6)/2, max_point+int(len(hq)*time_window/6)/2])
                img_name = f"{DATASET_PATH}/{event.split('-')[0]}_{detector}_{time_window}.png"
                plt.savefig(img_name, bbox_inches='tight', pad_inches=0)
                plt.close()
                if GRAY_SCALE:
                    Image.open(img_name).resize((170,140)).convert('L').save(img_name)