import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def plot_images(
    images,
    titles=None,
    save_path=None,
    rows=1,
    fontsize=20,
    figsize=(12, 3),
    show_axis=False,
):
    """
    Plotting images with titles
    """

    n_images = len(images)

    cols_per_row = [
        n_images // rows + (1 if i < n_images % rows else 0)
        for i in range(rows)
    ]
    total_cols = max(cols_per_row)

    fig, axs = plt.subplots(rows, total_cols, figsize=figsize)

    if rows == 1:
        axs = np.array([axs])
    if total_cols == 1:
        axs = axs.reshape(-1, 1)

    axs_flat = axs.ravel()

    image_idx = 0
    for i in range(rows):
        for j in range(total_cols):
            if j < cols_per_row[i] and image_idx < n_images:
                img = images[image_idx]

                if isinstance(img, torch.Tensor):
                    arr = img.detach().cpu().to(torch.float32).numpy()
                elif isinstance(img, np.ndarray):
                    arr = img.astype(np.float32)
                else:
                    arr = np.array(img, dtype=np.float32)

                arr = arr.squeeze()
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr.transpose(1, 2, 0)

                axs[i, j].imshow(arr, cmap="gray" if arr.ndim == 2 else None)
                if not show_axis:
                    axs[i, j].axis("off")
                if titles:
                    axs[i, j].set_title(
                        f"{titles[image_idx]}", fontsize=fontsize
                    )

                image_idx += 1
            else:
                axs[i, j].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    # plt.show()

def loss_plot(
    path2logs: str = "results/logs/CyclicMN.csv",
    savepath: str = "results/logs/loss.png",
    logscale: bool = False,
):
    """
    Plot loss descending using data from csv logs
    """
    df = pd.read_csv(path2logs)

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["EPOCH"],
        df["TRAIN LOSS"],
        "b-",
        label="Train Loss",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        df["EPOCH"],
        df["VALIDATION LOSS"],
        "r-",
        label="Validation Loss",
        linewidth=2,
        marker="s",
        markersize=4,
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(df["EPOCH"])
    if logscale:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(savepath)

def expand_zeros(x, shape: tuple[int, int]):
    h, w = x.shape
    new_h, new_w = shape

    new_x = np.zeros(shape)
    
    # top left corner
    c_x = (new_h - h) // 2
    c_y = (new_w - w) // 2

    new_x[c_x:c_x + h, c_y:c_y + w] = x
    return new_x

