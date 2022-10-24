"""Generate a (small) dataset with O3 glitches from Gravity Spy."""

import argparse
import os
import sys
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from gwpy.table import GravitySpyTable
from PIL import Image
from tqdm import tqdm

DATASET_PATH = "./datasets/GlitchesO3"

detectors = ["H1", "L1"]
obs_run = "O3b"
views = ["0.5", "1.0", "2.0", "4.0"]
img_size = (170, 140)  # (width, height)
# Pixel indexes of the spectrogram plot borders, to be used for cropping:
left, top, right, bottom = 105, 66, 671, 532


def choose_samples():
    """Choose the desired number of glitches for each class and detector.
    The gravityspy_ids and labels are saved in a meta_data file."""

    print("Selecting samples...")
    meta_data = pd.DataFrame(columns=["gravityspy_id", "ifo", "label"])
    for label in labels:
        for detector in detectors:
            detector_df = detector_dfs[detector]
            try:
                ids = np.random.choice(
                    detector_df.loc[detector_df["ml_label"] == label]["gravityspy_id"],
                    size=args.num_samples,
                    replace=False,
                )
            # When there are not enough different glitches, choose all of them.
            except ValueError:
                ids = np.array(detector_df.loc[detector_df["ml_label"] == label]["gravityspy_id"])
            # Save ids and labels to meta_data.
            for id_ in ids:
                meta_data.loc[len(meta_data)] = [id_, detector, label]

    meta_data.to_csv(meta_data_path, index=False)
    print("Sample selection saved.")


def generate_samples():
    """Download spectrograms from and generate 170x140 images for the events in meta_data file.
    Skips events which are already processed."""

    print("Saving images...")
    for row in tqdm(meta_data.values):
        id_, det, _ = row
        O3_data_row = O3_data[O3_data["gravityspy_id"] == id_]

        # Skip generation if image already in folder.
        last_img_name = f"{full_dset_path}/{det}_{id_}_spectrogram_{views[-1]}.png"
        if os.path.exists(last_img_name):
            continue

        # Try to download spectrograms from zenodo servers.
        try:
            O3_data_row.download(download_path=full_dset_path)
        except (HTTPError, URLError) as e:
            print(f"Download of samples with id {id_} failed.")
            files_to_remove = [
                f"{full_dset_path}/{det}_{id_}_spectrogram_{view}.png" for view in views
            ]
            for file_to_remove in files_to_remove:
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
            print("Trying again")
            O3_data_row.download(download_path=full_dset_path)

        # Crop the images to only include the spectrograms, convert to BW, and resize to 170x140.
        for view in views:
            img_name = f"{full_dset_path}/{det}_{id_}_spectrogram_{view}.png"
            crop_idxs = left, top, right, bottom
            Image.open(img_name).crop(crop_idxs).resize(img_size).convert("L").save(img_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_samples", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--ml_confidence", type=float, default=0.95)
    parser.add_argument("--dset_version", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    meta_data_path = f"{DATASET_PATH}/meta_data_O3_v{args.dset_version}.csv"
    full_dset_path = f"{DATASET_PATH}/GlitchesO3_v{args.dset_version}"

    O3_files = [f"{DATASET_PATH}/{detector}_{obs_run}.csv" for detector in detectors]
    O3_data = GravitySpyTable.read(O3_files)
    detector_dfs = {
        detector: O3_data.filter(
            f"ifo=={detector}", f"ml_confidence>={args.ml_confidence}"
        ).to_pandas()
        for detector in detectors
    }
    labels = sorted(np.unique(O3_data["ml_label"]))

    if args.new_samples:
        if not os.path.exists(full_dset_path):
            os.makedirs(full_dset_path)
        if os.path.exists(meta_data_path):
            print("Meta data file already exists. Aborting...")
            sys.exit(0)
        choose_samples()
    else:
        if not os.path.exists(meta_data_path):
            print("Meta data file does not exist. Aborting...")
            sys.exit(0)
        print("Loading previously chosen samples...")
    meta_data = pd.read_csv(meta_data_path)

    generate_samples()
