"""Utilities for the main notebooks.
Mainly plotting functions."""

import gc

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from fastai.basics import *
from fastai.vision.all import *
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator  # integer pyplot ticks
from sklearn import metrics


############  Dotted dictionaries ################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


############  Transformations ####################
def convert_to_3channel(x):
    x = torch.tensor(x, dtype=torch.float32)
    output = torch.stack([x, x, x])
    return output

def np_to_tensor(x):
    return torch.Tensor(x)

############  Get mean and std for each channel ############

def get_channels_stats(view, view_means, view_stds):
    if view == 'merged':
        means = np.mean(view_means)
        stds  = np.mean(view_stds)
    elif view.startswith('single'):
        means = view_means[int(view[-1])-1]
        stds  = view_stds[int(view[-1])-1]
    elif view.startswith('encoded'):
        means = [view_means[int(s_view[-1])-1] for s_view in view[7:]]
        stds  = [view_stds[int(s_view[-1])-1] for s_view in view[7:]]
    return means, stds

############  Plotting functions #################

def plot_loss(recorder):
    fig, ax = plt.subplots()
    epochs = list(range(len(recorder.values)))
    ax.plot(epochs, L(recorder.values).itemgot(0), label="train")
    ax.plot(epochs, L(recorder.values).itemgot(1), label="valid")
    ax.set_title("Learning Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross Entropy")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(epochs), integer=True))
    ax.legend()

    return fig, ax


def plot_metric(recorder):
    fig, ax = plt.subplots()
    epochs = list(range(len(recorder.values)))
    ax.plot(epochs, np.array(recorder.values)[:, 2])
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=len(epochs), integer=True))

    return fig, ax


def plot_PRC(ground_truth, preds):
    """plot precision-recall curves."""
    glitch_list = list(ds_test.class_dict.keys())
    precision, recall, average_precision = {}, {}, {}

    for i in range(len(glitch_list)):
        precision[i], recall[i], _ = metrics.precision_recall_curve(
            ground_truth[:, i], preds[:, i]
        )
        average_precision[i] = metrics.average_precision_score(
            ground_truth[:, i], preds[:, i], average="weighted"
        )

    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(
        ground_truth.ravel(), preds.ravel()
    )
    average_precision["micro"] = metrics.average_precision_score(
        ground_truth, preds, average="micro"
    )

    # setup plot details
    colors = list(mcolors.CSS4_COLORS)[::4]
    for rm in ["navajowhite", "white", "azure", "linen", "coral", "blanchedalmond"]:
        colors.remove(rm)

    fig = plt.figure(figsize=[16, 10])
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall["micro"], precision["micro"], color="gold", lw=2)
    lines.append(l)
    labels.append(
        "Micro-Averaged Precision-Recall (area = {0:0.2f})" "".format(average_precision["micro"])
    )

    for i, color in zip(range(22), colors):
        (l,) = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(
            "Precision-Recall for class {0} (area = {1:0.2f})"
            "".format(glitch_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.title("Precision-Recall curve across all classes", fontsize=18)
    ax.legend(lines, labels, bbox_to_anchor=(1.001, 1), loc=2, prop=dict(size=14))

    return fig, ax






################# Predictions functions #############


def get_preds(learner, ds_test):
    dl_test = learner.dls.test_dl(ds_test, with_labels=True)
    preds, targets = learner.get_preds(dl=dl_test)
    soft_preds = F.softmax(preds, dim=1)
    y_true = [np.argmax(target) for target in targets]
    y_pred = [np.argmax(pred) for pred in preds]

    return preds, targets, soft_preds, y_true, y_pred


############# Run model multiple times #############


class CompareModels:
    def __init__(
        self,
        dls,
        architecture="resnet34",
        num_classes=22,
        transfer_learning=False,
        max_lr=steep,
        max_lr_multiplier=1,
        num_epochs=20,
        freeze_epochs=1,
        fine_tune_lr_mult=100,
        mixed_precision=False,
    ):
        self.dls = dls
        self.architecture = architecture
        self.num_classes = num_classes
        self.max_lr = max_lr
        self.lr_multiplier = max_lr_multiplier
        self.num_epochs = num_epochs
        self.freeze_epochs = freeze_epochs
        self.lr_mult = fine_tune_lr_mult
        self.transfer_learning = transfer_learning
        self.mixed_precision = mixed_precision

        self.max_scores = []
        self.best_epochs = []
        self.training_time = 0

    def run_training(self, n_runs=5):
        for run in range(n_runs):
            gc.collect()
            self._init_model(run)
            clear_output()
            print(
                f"Training model #{run+1}/{n_runs} (avg best score so far: {np.mean(self.max_scores):.5f})"
            )
            start_time = time.time()
            self._train()
            end_time = time.time()
            self.training_time += (end_time - start_time) / n_runs
        print("\nBest scores:")
        print(*self.max_scores, sep="\n", end="\n")

    def print_stats(self):
        print(f"best epochs: {self.best_epochs} (avg: {np.mean(self.best_epochs):.1f})")
        print(f"SCORE:\tmean: {np.mean(self.max_scores):.5f}, std: {np.std(self.max_scores):.5f}")

    def save_model_info(self, name, img_augs=False, path="experiments.csv"):
        model_architecture = self.architecture.split("_")[0]
        num_channels = 1 if self.architecture.endswith("1ch") else 3
        avg_best_epochs = (
            np.mean(self.best_epochs) + 1 + self.freeze_epochs
            if self.transfer_learning
            else np.mean(self.best_epochs) + 1
        )
        lr = (
            f"{mul}*{(self.max_lr).__name__}"
            if (mul := self.lr_multiplier) != 1
            else (self.max_lr).__name__
        )
        if self.transfer_learning:
            info = (
                f"\n{name}, {model_architecture}, {self.dls.train_ds.view}, {self.transfer_learning}, {num_channels}"
                + f", {img_augs}, {self.freeze_epochs} , {self.num_epochs}, {lr}"
                + f", {np.mean(self.max_scores):.5f}, {avg_best_epochs:.1f}, {int(self.training_time)}"
            )
        else:
            img_size = "x".join([str(x) for x in list(self.dls.one_batch()[0].shape[-2:])])
            info = (
                f"\n{name}, {model_architecture}, {self.transfer_learning}, {self.dls.train_ds.view}, {num_channels}"
                + f", {img_size}, {self.dls.bs} , {self.num_epochs}, {lr}"
                + f", {np.mean(self.max_scores):.5f}, {avg_best_epochs:.1f}, {int(self.training_time)}"
            )
        with open(path, "a") as f:
            f.write(info)

    def _init_model(self, run):
        metrics = [accuracy_multi, F1ScoreMulti(average="macro")]
        if self.transfer_learning:
            if self.architecture == "resnet34":
                self.model = models.resnet34
            elif self.architecture == "resnet18":
                self.model = models.resnet18
            elif self.architecture == "resnet50":
                self.model = models.resnet50

            self.learner = vision_learner(
                self.dls,
                self.model,
                n_out=self.num_classes,
                loss_func=nn.BCEWithLogitsLoss(),
                metrics=metrics,
            )
        else:
            if self.architecture.startswith("resnet34"):
                self.model = models.resnet34(num_classes=self.num_classes)
            elif self.architecture.startswith("resnet18"):
                self.model = models.resnet18(num_classes=self.num_classes)
            elif self.architecture.startswith("resnet50"):
                self.model = models.resnet50(num_classes=self.num_classes)

            # Change first layer if only one channel is desired.
            if self.architecture.endswith("1ch"):
                self.model.conv1 = nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )

            self.learner = Learner(
                self.dls,
                self.model,
                loss_func=nn.BCEWithLogitsLoss(),
                metrics=metrics,
            )

        if self.mixed_precision:
            self.learner.to_fp16()

        show_plot = True if run == 0 else False
        self.get_lr = self.learner.lr_find(suggest_funcs=self.max_lr, show_plot=show_plot)

    def _train(self):
        if not self.transfer_learning:
            self.learner.fit_one_cycle(self.num_epochs, lr_max=self.lr_multiplier * self.get_lr[0])
        else:
            self.learner.fine_tune(
                self.num_epochs,
                base_lr=self.lr_multiplier * self.get_lr[0],
                freeze_epochs=self.freeze_epochs,
                lr_mult=self.lr_mult,
            )

        self.max_scores.append(max(np.array(self.learner.recorder.values)[:, -1]))
        self.best_epochs.append(np.argmax(np.array(self.learner.recorder.values)[:, -1]))

