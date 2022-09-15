from fastai.basics import *
from torch.utils.data import Dataset
from PIL import Image

class Data_GW(Dataset):
    def __init__(self, dataset_path, view="encoded134", one_hot=False, transform=None):
        self.dataset_path = Path(dataset_path)
        files = self.dataset_path.glob('*.png')
        self.events = sorted(set([file.name.rpartition('_')[0] for file in files]))
        
        self.transform = transform

        if one_hot:
            self.class_dict = {
                key: F.one_hot(torch.tensor(i), num_classes=22).float()
                for i, key in enumerate([None] * 4 + ['Chirp'] + [None] * 17)
            }
        else:
            self.class_dict = {
                key: i
                for i, key in enumerate([None] * 4 + ['Chirp'] + [None] * 17)
            }
        
        self.view_dict = {"1": "0.5.png", "2": "1.0.png", "3": "2.0.png", "4": "4.0.png"}
        assert view == 'merged' or (
            view.startswith('single') and view[-1] in self.view_dict.keys()
        )  or (
            view.startswith("encoded") and len(view) >= len("encoded") + 2
        ), "wrong view format"
        self.view = view

    def __getitem__(self, idx):
        label = "Chirp"
        view = self.view
        event = self.events[idx]
        
        if self.view == "merged":
            view1 = Image.open(self.dataset_path/f'{event}_0.5.png')
            view2 = Image.open(self.dataset_path/f'{event}_1.0.png')
            view3 = Image.open(self.dataset_path/f'{event}_2.0.png')
            view4 = Image.open(self.dataset_path/f'{event}_4.0.png')
            img = np.vstack((np.hstack((view1, view2)), (np.hstack((view3, view4)))))
        elif self.view.startswith("encoded"):
            n_views = len(self.view) - len('encoded')
            chosen_views = list(self.view[-n_views:])
            channels = [np.asarray(Image.open(self.dataset_path/f'{event}_{self.view_dict[chosen_view]}'))/255. for chosen_view in chosen_views]
            img = np.array(channels)
        else:
            img = Image.open(self.dataset_path/f'{event}_{self.view_dict[view[-1]]}')

        if self.transform:
            img = self.transform(img)
                
        return (img, self.class_dict[label])

    def __len__(self):
        return len(self.events)
