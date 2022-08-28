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

# Dictionary with the implemented models.
# zero_init_last is the default torchvision behaviour. 
# It yields better results when trianing fewer epochs.
model_dict = {'resnet18': partial(models.resnet18, zero_init_last=False),
              'resnet26': partial(models.resnet26, zero_init_last=False),
              'resnet34': partial(models.resnet34, zero_init_last=False),
              'resnet50': partial(models.resnet50, zero_init_last=False),
              'resnet101': partial(models.resnet101, zero_init_last=False),
              'convnext_nano': models.convnext_nano,
              'convnext_tiny': models.convnext_tiny,
              'convnext_small': models.convnext_small,
              'efficientnetv2_rw_t': models.efficientnetv2_rw_t,
              'efficientnetv2_s': models.efficientnetv2_s
             }

# Dictionary with the available lr suggestion methods.
sug_func_dict = {"minimum": minimum, 
                 "valley": valley, 
                 "steep": steep, 
                 "slide": slide}

def get_dls(config):
    """Create the DataLoaders given a configuration.
    Also returns the image size and number of channels."""
    
    # Get size of the image lowest dimension.
    image_dim = config.get('image_dim', 140)
    
    # Interpolate size if merged view.
    if config.view == "merged":
        image_dim = int(interp1d([50,140],[50,280])(image_dim))
    
    # Get width and height for square and non-square images.
    if config.get('image_square', False):
        image_size = (image_dim, image_dim)
    else:
        image_size = (image_dim, int(image_dim * (170 / 140)))

    # Determine number of channels.
    if config.view.startswith("encoded"):
        n_channels = len(config.view) - len("encoded")
    else:
        n_channels = config.get('n_channels', 1)

    # Use appropriate transform for encoded views and single views with 3 channels.
    if config.view.startswith("encoded"):
        train_transforms = [np_to_tensor]
    elif n_channels == 3:
        train_transforms = [convert_to_3channel]
    else:
        train_transforms = [tfms.ToTensor()]
    train_transforms.append(tfms.Resize(image_size))
    valid_transforms = train_transforms.copy()
    
    # Add re-scaling and shifts to the training transformations.
    tfm_zoom_range = config.get('tfm_zoom_range', 0.0)
    tfm_shift_fraction = config.get('tfm_shift_fraction', 0.0)
    if tfm_zoom_range or tfm_shift_fraction:
        scale = (1 - tfm_zoom_range, 1 + tfm_zoom_range) if tfm_zoom_range else None
        translate = (tfm_shift_fraction, tfm_shift_fraction) if tfm_shift_fraction else None
        shift_and_zoom = tfms.RandomAffine(degrees=0, translate=translate, scale=scale)
        train_transforms.append(shift_and_zoom)
    
    train_transforms = tfms.Compose(train_transforms)
    valid_transforms = tfms.Compose(valid_transforms)
    
    # Check if dataset with changed labels is to be used.
    correct_labels = config.get('correct_labels', False)

    ds = Data_GW(
        dataset_path=DATASET_PATH, data_type="train", view=config.view, correct_labels=correct_labels, transform=train_transforms
    )
    ds_val = Data_GW(
        dataset_path=DATASET_PATH, data_type="validation", view=config.view, correct_labels=correct_labels, transform=valid_transforms
    )

    dls = DataLoaders.from_dsets(ds, ds_val, bs=config.batch_size, device=device)

    return dls, image_size, n_channels              

def get_learner(config, dls, n_channels):
    """Create the Learner given a configuration and a DataLoaders."""
    
    # Training metrics.
    metrics = [accuracy, F1Score(average="macro")]
    
    # Get re-weighting tensor.
    weighted_loss = config.get('weighted_loss', False)
    if weighted_loss:
        filt = dls.train_ds.meta_data['sample_type'].isin(['train', 'validation'])
        samples_per_class = tensor(dls.train_ds.meta_data.loc[filt]['label'].value_counts(sort=False))
        
        if weighted_loss == 'inverse':
            class_weights = 1.0 / samples_per_class
        elif weighted_loss == 'effective':
            beta = config.get('weighted_loss_beta', 0.99)
            class_weights = (1.0 - beta) / (1.0 - torch.pow(beta, samples_per_class))
        
        class_weights = class_weights / torch.sum(class_weights) * len(class_weights)
        
        # Move weights to GPU:
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    # Get the desired loss function.
    focal_loss = config.get('focal_loss', False)
    label_smoothing = config.get('label_smoothing', False)
    if not (label_smoothing or focal_loss):
        loss_func = CrossEntropyLossFlat(weight=class_weights)
    elif (label_smoothing and focal_loss):
        raise NotImplementedError('Simulatenous use of focal loss and label smoothing is not supported.')
    elif label_smoothing:
        loss_func = LabelSmoothingCrossEntropyFlat(weight=class_weights, eps=label_smoothing)
    elif focal_loss:
        gamma = config.get('focal_loss_gamma', 2.0)
        loss_func = FocalLossFlat(weight=class_weights, gamma=gamma)
    
    
    cbs = [ShowGraphCallback()]
    
    # Mixup.
    if config.get('mixup', False):
        cbs.append(MixUp(alpha=config.mixup))
    
    # Avoid using wandb callback if not training the model.
    if config.get('inference', False) == False:
        cbs.append(WandbCallback(log_model=False, log_preds=False))

    # Get the desired architecture.
    model = model_dict[config.architecture](in_chans=n_channels, num_classes=n_classes,
                                            pretrained=config.transfer_learning)
    learner = Learner(
        dls, model, loss_func=loss_func, metrics=metrics, cbs=cbs
    )

    if config.mixed_precision:
        learner.to_fp16()

    return learner


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Clean memory to avoid errors.
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get DataLoaders and Learner.
        dls, image_size, n_channels = get_dls(config=config)
        learner = get_learner(config=config, dls=dls, n_channels=n_channels)

        # Get the learning rate.
        sug_func = sug_func_dict[config.suggest_func]
        get_lr = learner.lr_find(suggest_funcs=sug_func)
        lr = config.get('lr_multiplier', 1) * get_lr[0]
        wandb.summary["lr_finder_lr"] = lr
        
        # Train the model.
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
        
        # Save model after training if desired.
        # with_opt=False produces much smaller files, but can only be used for inference.
        if config.get('save_model', False):
            file_name = f"{sweep_id.split('/')[-1]}_run{config.experiment_no}"
            learner.save(file_name, with_opt=False)
        
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
        type=int,
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
