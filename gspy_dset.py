import h5py
from fastai.basics import *
from torch.utils.data import Dataset


class Data_Glitches(Dataset):
    def __init__(
        self,
        dataset_path,
        data_type="train",
        view="encoded134",
        one_hot=False,
        correct_labels=False,
        transform=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.correct_labels = correct_labels
        self.meta_data = pd.read_csv(self.dataset_path / "trainingset_v1d0_metadata.csv")
        if one_hot:
            self.class_dict = {
                key: F.one_hot(torch.tensor(i), num_classes=22).float()
                for i, key in enumerate(np.unique(self.meta_data["label"]))
            }
        else:
            self.class_dict = {key: i for i, key in enumerate(np.unique(self.meta_data["label"]))}
        assert data_type in self.meta_data["sample_type"].unique(), "unknown data_type"
        self.data_type = data_type
        self.view_dict = {"1": "0.5.png", "2": "1.0.png", "3": "2.0.png", "4": "4.0.png"}
        assert (
            view == "merged"
            or (view.startswith("single") and view[-1] in self.view_dict.keys())
            or (view.startswith("encoded") and len(view) >= len("encoded") + 2)
        ), "wrong view format"
        self.view = view

        filt = self.meta_data["sample_type"] == self.data_type
        self.samples = self.meta_data.loc[filt, ["gravityspy_id", "label"]]

    def __getitem__(self, idx):
        label = self.samples.iloc[idx]["label"]
        grp = self.samples.iloc[idx]["gravityspy_id"]
        data_type = self.data_type
        view = self.view
        with h5py.File(self.dataset_path / "trainingsetv1d0.h5", "r") as f:
            if self.view == "merged":
                view1 = f[label][data_type][grp]["0.5.png"][0]
                view2 = f[label][data_type][grp]["1.0.png"][0]
                view3 = f[label][data_type][grp]["2.0.png"][0]
                view4 = f[label][data_type][grp]["4.0.png"][0]
                img = np.vstack((np.hstack((view1, view2)), (np.hstack((view3, view4)))))
            elif self.view.startswith("encoded"):
                n_views = len(self.view) - len("encoded")
                chosen_views = list(self.view[-n_views:])
                channels = [
                    f[label][data_type][grp][self.view_dict[chosen_view]][0]
                    for chosen_view in chosen_views
                ]
                img = np.array((channels))
            else:
                img = f[label][data_type][grp][self.view_dict[view[-1]]][0]

        if self.transform:
            img = self.transform(img)

        # Correct wrongly labelled sample.
        if self.correct_labels and grp == "VXw4KBWl5g":
            label = "Paired_Doves"

        return (img, self.class_dict[label])

    def __len__(self):
        return len(self.samples)
