import numpy as np
from fastai.basics import *
from fastai.vision.all import *
from matplotlib import gridspec
from sklearn import metrics


def get_preds(learner, ds_idx=1):
    """Get predictions and targets."""
    preds, targets = learner.get_preds(ds_idx)
    soft_preds = F.softmax(preds.float(), dim=1)
    if targets.ndim > 1:
        y_true = [np.argmax(target) for target in targets]
    else:
        y_true = targets
    y_pred = [np.argmax(pred) for pred in preds]
    return preds, targets, soft_preds, y_true, y_pred


def get_val_preds(learner):
    """Deprecated!! Use get_preds instead.
    Get predictions and targets for the validation set."""
    return get_preds(learner)

def plot_CM_PR(cm, y_true, y_pred, vocab, normalize=False, figsize=(10, 10)):
    """Plot confusion matrix with precision and recall for each class at the bottom,"""
    if normalize:
        assert any(y_true) and any(y_pred), "predicted labels or targets missing"
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])

    ax0 = plt.subplot(gs[0])
    plt.imshow(cm, interpolation="nearest", cmap="Blues", aspect=0.85)
    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=45, ha="left", fontsize="large")
    plt.yticks(tick_marks, vocab, fontsize="large")

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.005:
            coeff = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax = plt.gca()
    ax.tick_params(axis="x", length=0, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.set_label_position("right")
    plt.ylabel("Actual Class", labelpad=10, fontsize="x-large")

    ax1 = plt.subplot(gs[1])
    prec = metrics.precision_score(y_true, y_pred, average=None)
    rec = metrics.recall_score(y_true, y_pred, average=None)
    pr = np.vstack((rec, prec))
    x_vocab = vocab
    y_vocab = ["Recall", "Precision"]
    plt.imshow(pr, interpolation="nearest", cmap="Blues", vmin=0, vmax=1, aspect=1)
    x_tick_marks = np.arange(len(x_vocab))
    y_tick_marks = np.arange(len(y_vocab))
    plt.yticks(y_tick_marks, y_vocab, fontsize="large")
    thresh = pr.max() / 2.0
    for i, j in itertools.product(range(pr.shape[0]), range(pr.shape[1])):
        if pr[i, j] >= 0.005:
            coeff = f"{pr[i, j]:.2f}"
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if pr[i, j] > thresh else "black",
                weight="bold",
                size=9.5,
            )
    ax = plt.gca()
    ax.tick_params(axis="x", length=0, top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.yaxis.set_tick_params(length=0)
    plt.xlabel("Predicted Class", labelpad=10, fontsize="x-large")
    plt.grid(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, [ax0, ax1]

def plot_CM_chirp_line(ax, cm, vocab=None, normalize=False):
    """Plot the Chirp line from the confusion matrix."""
    im = ax.imshow(cm[4].reshape(1, 22), interpolation="nearest", cmap="Blues", aspect=0.85 if normalize else 1)

    tick_marks = np.arange(len(vocab))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(vocab, rotation=45, ha="left", fontsize="large")
    ax.set_yticks([0])
    ax.set_yticklabels([vocab[4]], fontsize="large")
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.005:
            coeff = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(
                j,
                0,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.tick_params(axis="x", length=0, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Actual Class", labelpad=30, fontsize="x-large")
    ax.set_xlabel("Predicted Class", labelpad=10, fontsize="x-large")
    ax.grid(False)

    return ax

def plot_top_losses_glitches(
    interp,
    learner,
    ds_idx=1,
    y_preds=None,
    largest=True,
    vocab=None,
    channel_list=None,
    show_label=True,
    show_pred=True,
    show_loss=False,
    nrows=4,
    ncols=4,
    figsize=(11, 11),
):
    """Plot top losses for the Gravity Spy dataset."""
    view_dict = {0: "0.5.png", 1: "2.0.png", 2: "4.0.png"}

    ds = learner.dls.loaders[ds_idx].dataset

    top_losses = interp.top_losses(nrows * ncols, largest=largest)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    for i, idx in enumerate(top_losses[1]):
        ax = axes.flat[i]
        idx = int(idx)
        true_label = vocab[ds[idx][1]]
        channel = channel_list[i] if i < len(channel_list) else 2
        view_time = float(view_dict[channel][:3])
        freq_pos = np.linspace(-100, 2048, 9)[1:-1]
        freqs = np.logspace(3, 11, num=9, base=2)[1:-1]
        times = np.linspace(-view_time / 2, view_time / 2, 5)

        img = ax.imshow(
            ds[idx][0][channel],
            extent=[-view_time / 2, view_time / 2, 8, 2048],
            aspect=140 / 170 * view_time / 2038,
        )
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticks(times, [f"{float(time)}" for time in times])
        ax.set_yticks(freq_pos, [f"{freq:.0f}" for freq in freqs])
        cbar = plt.colorbar(img, ax=ax, shrink=0.7)
        if (i + 1) % ncols == 0:
            cbar.set_label("Normalized energy")

        title = ""
        if show_label:
            title += f"label: {true_label}"
        if show_pred:
            title += f"\npred: {vocab[y_preds[idx]]}"
        if show_loss:
            title += f"\nloss: {top_losses[0][i+first_idx]:.2e}"
        if title:
            ax.set_title(title)

    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (Hz)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time(s)")
    fig.tight_layout()

    return fig, axes


def plot_top_losses_gws(
    interp,
    learner,
    ds_idx=2,
    y_preds=None,
    vocab=None,
    largest=True,
    show_label=True,
    show_pred=True,
    show_loss=False,
    nrows=4,
    ncols=4,
    figsize=(11, 11),
):
    """Plot top losses for the real GW events dataset."""
    ds = learner.dls.loaders[ds_idx].dataset
    top_losses = interp.top_losses(nrows * ncols, largest=largest)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    for i, idx in enumerate(top_losses[1]):
        ax = axes.flat[i]
        idx = int(idx)
        true_label = vocab[ds[idx][1]]
        freq_pos = np.linspace(-100, 2048, 9)[1:-1]
        freqs = np.logspace(3, 11, num=9, base=2)[1:-1]
        view_time = 4
        times = np.linspace(-view_time / 2, view_time / 2, 5)

        img = ax.imshow(
            ds[idx][0].permute(1, 2, 0),
            extent=[-view_time / 2, view_time / 2, 8, 2048],
            aspect=140 / 170 * view_time / 2038,
        )
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_yticks(freq_pos, [f'{freq:.0f}' for freq in freqs]);

        title = [r"$\bf{" + ds.events[idx].replace("_", " - ") + "}$"]
        if show_label:
            title.append(f"label: {true_label}")
        if show_pred:
            title.append(f"pred: {vocab[y_preds[idx]]}")
        if show_loss:
            title.append(f"loss: {top_losses[0][i+first_idx]:.2e}")
        if title:
            ax.set_title("\n".join(title))

    # for ax in axes[:,0]: ax.set_ylabel('Frequency (Hz)')
    fig.tight_layout()

    return fig, axes


def plot_CM(cm, vocab=None, normalize=False, y_true=None, y_pred=None, figsize=(12, 12)):
    """Plot confusion matrix."""

    def _add_to_matrix(cm):
        prec = metrics.precision_score(y_true, y_pred, average=None)
        rec = metrics.recall_score(y_true, y_pred, average=None)

        return np.vstack((cm, rec, prec))

    if normalize:
        assert any(y_true) and any(y_pred), "predicted labels or targets missing"
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = _add_to_matrix(cm)
        extended_vocab = vocab + ["Recall", "Precision"]

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap="Blues", aspect=0.85 if normalize else 1)

    tick_marks = np.arange(len(vocab))
    plt.xticks(tick_marks, vocab, rotation=45, ha="left", fontsize="large")
    if normalize:
        extended_tick_marks = np.arange(len(extended_vocab))
        plt.yticks(extended_tick_marks, extended_vocab, fontsize="large")
    else:
        plt.yticks(tick_marks, vocab, fontsize="large")
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] >= 0.005:
            coeff = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax = plt.gca()
    ax.tick_params(axis="x", length=0, top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.yaxis.set_tick_params(length=0)
    ax.yaxis.set_label_position("right")
    plt.tight_layout()
    plt.ylabel("Actual Class", labelpad=10, fontsize="x-large")
    plt.xlabel("Predicted Class", labelpad=10, fontsize="x-large")
    plt.grid(False)

    return fig, ax


def plot_PR_extension(y_true=None, y_pred=None, vocab=None):
    prec = metrics.precision_score(y_true, y_pred, average=None)
    rec = metrics.recall_score(y_true, y_pred, average=None)
    pr = np.vstack((rec, prec))
    y_vocab = ["Recall", "Precision"]
    x_vocab = vocab
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(pr, interpolation="nearest", cmap="Blues", vmin=0, vmax=1, aspect=1)
    x_tick_marks = np.arange(len(x_vocab))
    y_tick_marks = np.arange(len(y_vocab))
    plt.yticks(y_tick_marks, y_vocab, fontsize="large")
    thresh = pr.max() / 2.0
    for i, j in itertools.product(range(pr.shape[0]), range(pr.shape[1])):
        if pr[i, j] >= 0.005:
            coeff = f"{pr[i, j]:.2f}"
            plt.text(
                j,
                i,
                coeff,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if pr[i, j] > thresh else "black",
            )
    ax = plt.gca()
    ax.tick_params(axis="x", length=0, top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.yaxis.set_tick_params(length=0)
    plt.xlabel("Predicted Class", labelpad=10, fontsize="x-large")
    plt.grid(False)
    plt.tight_layout()

    return fig, ax
