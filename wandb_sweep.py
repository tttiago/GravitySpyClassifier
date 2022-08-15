import argparse
import gc
import json
import time
import warnings

import torchvision.transforms as tfms
import wandb
from fastai.basics import *
from fastai.callback.wandb import *
from fastai.vision.all import *
from scipy.interpolate import interp1d
from timm import models

warnings.filterwarnings("ignore")

from gspy_dset import Data_GW
from my_utils import convert_to_3channel, np_to_tensor

#####################################################################

DATASET_PATH = "./datasets/Glitches"
PROJECT = 'thesis_gravity_spy'
device = "cuda" if torch.cuda.is_available() else "cpu"
n_classes = 22


def get_dls(config):
    image_dim = (
        config.image_dim if config.view != "merged" else int(interp1d([50,140],[50,280])(config.image_dim))
    )
    if config.image_square == True:
        image_size = (image_dim, image_dim)
    else:
        image_size = (image_dim, int(image_dim * (170 / 140)))

    n_channels = config.get('n_channels', 1)
    if config.view.startswith("encoded"):
        n_channels = len(config.view) - len("encoded")

    if config.view.startswith("enc"):
        train_transforms = [np_to_tensor]
    elif n_channels == 3:
        train_transforms = [convert_to_3channel]
    else:
        train_transforms = [tfms.ToTensor()]
    train_transforms.append(tfms.Resize(image_size))
    valid_transforms = train_transforms.copy()
        
    try:    
        if config.tfm_zoom_range or config.tfm_shift_fraction:
            scale = (1 - config.tfm_zoom_range, 1 + config.tfm_zoom_range) if config.tfm_zoom_range else None
            translate = (config.tfm_shift_fraction, config.tfm_shift_fraction) if config.tfm_shift_fraction else None
            shift_and_zoom = tfms.RandomAffine(degrees=0, translate=translate, scale=scale)
            train_transforms.append(shift_and_zoom)
    except (AttributeError, KeyError):
        pass
    
    correct_labels = config.get('correct_labels', False)
    
    train_transforms = tfms.Compose(train_transforms)
    valid_transforms = tfms.Compose(valid_transforms)

    ds = Data_GW(
        dataset_path=DATASET_PATH, data_type="train", view=config.view, correct_labels=correct_labels, transform=train_transforms
    )
    ds_val = Data_GW(
        dataset_path=DATASET_PATH, data_type="validation", view=config.view, correct_labels=correct_labels, transform=valid_transforms
    )

    dls = DataLoaders.from_dsets(ds, ds_val, bs=config.batch_size, device=device)

    return dls, image_size, n_channels


# zero_init_last is the torchvision behaviour. yields better results when trianing fewer epochs.
model_dict = {'resnet18': partial(models.resnet18, zero_init_last=False),
              'resnet26': partial(models.resnet26, zero_init_last=False),
              'resnet34': partial(models.resnet34, zero_init_last=False),
              'resnet50': partial(models.resnet50, zero_init_last=False),
              'convnext_nano': models.convnext_nano,
              'convnext_tiny': models.convnext_tiny,
              'convnext_small': models.convnext_small,
             }
              

def get_learner(config, dls, n_channels):
    metrics = [accuracy, F1Score(average="macro")]
    
    if config.get('weighted_loss', False):
        filt = dls.train_ds.meta_data['sample_type'].isin(['train', 'validation'])
        samples_per_class = dict(dls.train_ds.meta_data.loc[filt]['label'].value_counts())
        class_weights = tensor([max(samples_per_class.values())/n_samples for _, n_samples in sorted(samples_per_class.items())])
    else:
        class_weights = None
    
    label_smoothing = config.get('label_smoothing', False)
    #loss_func = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if not label_smoothing:
        loss_func = CrossEntropyLossFlat(weight=class_weights)
    else:
        loss_func = LabelSmoothingCrossEntropyFlat(weight=class_weights, eps=label_smoothing)
    
    cbs = [ShowGraphCallback()]
        
    if config.get('mixup', False):
        cbs.append(MixUp(alpha=config.mixup))
    
    if config.get('inference', False) == False:
        cbs.append(WandbCallback(log_model=False, log_preds=False))
    
    if config.get('save_model', False):
        file_name = f"{sweep_id.split('/')[-1]}_run{config.experiment_no}"
        cbs.append(SaveModelCallback(fname=file_name, monitor='f1_score'))


    model = model_dict[config.architecture](in_chans=n_channels, num_classes=n_classes,
                                            pretrained=config.transfer_learning)
    learner = Learner(
        dls, model, loss_func=loss_func, metrics=metrics, cbs=cbs
    )

    if config.mixed_precision:
        learner.to_fp16()

    return learner


sug_func_call_dict = {"minimum": minimum, "valley": valley, "steep": steep, "slide": slide}


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        gc.collect()
        torch.cuda.empty_cache()

        dls, image_size, n_channels = get_dls(config=config)
        learner = get_learner(config=config, dls=dls, n_channels=n_channels)
        # lr = config.learning_rate
        sug_func = sug_func_call_dict[config.suggest_func]
        get_lr = learner.lr_find(suggest_funcs=sug_func)
        lr = config.get('lr_multiplier', 1) * get_lr[0]
        wandb.summary["lr_finder_lr"] = lr
        
        start_time = time.perf_counter()
        if not config.transfer_learning:
            learner.fit_one_cycle(config.epochs, lr_max=lr)
        else:
            learner.fine_tune(
                config.epochs,
                base_lr=lr,
                freeze_epochs=config.frozen_epochs,
            )
        
        total_runtime = time.perf_counter() - start_time
        best_f1 = max(np.array(learner.recorder.values)[:, -1])
        combined_f1_time = np.array(learner.recorder.values)[-1, -1] - total_runtime / 30000
        combined_bestf1_time = best_f1 - total_runtime / 30000
        wandb.summary["image_size"] = image_size
        wandb.summary["total_runtime"] = total_runtime
        wandb.summary["best_f1"] = best_f1
        wandb.summary["combined_f1_time"] = combined_f1_time
        wandb.summary["combined_bestf1_time"] = combined_bestf1_time
        

def parse_args():
    parser = argparse.ArgumentParser(description="Perform WandB sweep.")
    parser.add_argument(
        "-i",
        "--sweep_id",
        metavar="SID",
        type=str,
        nargs="?",
        help="ID of an existing sweep.",
    )
    parser.add_argument(
        "-f",
        "--config_file",
        metavar="CF",
        type=str,
        nargs="?",
        default="./sweep_config.json",
        help="Config file to create a new sweep.",
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        metavar="N",
        nargs="?",
        type=str,
        help="Number of runs in the sweep (optional).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    with open('wandb_key.txt', 'r') as f:
        wandb_key = f.read()
    wandb.login(key=wandb_key)

    args = parse_args()

    if args.sweep_id:
        sweep_id = f"ogait/{PROJECT}/{args.sweep_id}"
    elif args.config_file:
        with open(args.config_file) as f:
            sweep_config = json.load(f)
        sweep_id = wandb.sweep(sweep_config, project=PROJECT)

    if args.num_runs:
        wandb.agent(sweep_id, function=train, count=args.num_runs)
    else:
        wandb.agent(sweep_id, function=train)
