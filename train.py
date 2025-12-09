import argparse
import os

import csv
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.nn as nn

from models.unet import UNetSameSize
from dataset import getDenoiseLoader

from utils.stuff import loss_plot, plot_images

def linear_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def auto_create_path(FilePath):
    if os.path.exists(FilePath):
        print(f"[Info]: {FilePath} exists!", flush=True)
    else:
        print(f"[Info]: {FilePath} not exists!", flush=True)
        os.makedirs(FilePath)

def train(opt, model):
    # set model
    model.train()
    model.to(opt.device)

    # set dataset and dataloader
    train_dataloader, val_dataloader, _ = getDenoiseLoader(
        "C:/Users/user/Desktop/denoise/data/images/sca2023", 
        "C:/Users/user/Desktop/denoise/data/psf/levin",
        opt.imgs_per_batch,
        opt.batchsize,
        opt.shuffle,
        opt.val_ratio,
        opt.test_ratio,
        opt.seed
    )

    print(
        f"[Info]: Finish loading data! You have {len(train_dataloader) * opt.batchsize} images for train and {len(val_dataloader) * opt.batchsize} for validation"
    )

    save_loss_dir = os.path.join(
        opt.save_model_path, "logs", f"{opt.model_name}.csv"
    )
    save_plot_dir = os.path.join(
        opt.save_model_path, f"logs/loss_{opt.model_name}.png"
    )

    with open(save_loss_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["EPOCH", "TRAIN LOSS", "VALIDATION LOSS"])

    # optimizer
    optimizer_model = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        amsgrad=True,
        weight_decay=opt.regularization,
    )

    loss = torch.nn.MSELoss()
    # sigmoid = torch.nn.Sigmoid()

    print("[Info]: Start training!", flush=True)
    for epoch in range(opt.max_epochs):
        print(f"[epoch]: {epoch}")

        tot_val_loss = list()
        tot_train_loss = list()

        for idx, (imgs, gt) in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
        ):
            optimizer_model.zero_grad()

            imgs = imgs.to(opt.device)
            gt = gt.to(opt.device)

            # model output
            outputs = model(imgs)

            # loss
            denoise_loss = loss(outputs, gt)

            denoise_loss.backward()
            tot_train_loss.append(denoise_loss.item())
            optimizer_model.step()

            if idx % 1000 == 0:
                imgs2show = [imgs[0][3*k:3*k+3]for k in range(opt.imgs_per_batch)]
                imgs2show.append(gt[0])
                imgs2show.append(outputs[0])

                titles = [""] * opt.imgs_per_batch + ["GT Source"] + ["Model out"]
                plot_images(imgs2show,
                            titles,
                            figsize=(20, 20),
                            fontsize=20,
                            save_path=f"results/examples/batch_{idx}.png")   

        with torch.no_grad():
            for idx, (imgs, gt) in tqdm(
                enumerate(val_dataloader),
                desc="Validation",
                total=len(val_dataloader),
            ):

                imgs = imgs.to(opt.device)
                gt = gt.to(opt.device)

                # model output
                outputs = model(imgs)

                # Calculate loss
                denoise_loss = loss(outputs, gt)

                tot_val_loss.append(denoise_loss.item())  

                # if idx % 1000 == 0:
                #     imgs2show = [imgs[0][3*k:3*k+3]for k in range(opt.imgs_per_batch)]
                #     imgs2show.append(gt[0])
                #     imgs2show.append(outputs[0])

                #     titles = [""] * opt.imgs_per_batch + ["GT Source"] + ["Model out"]
                #     plot_images(imgs2show,
                #                 titles,
                #                 figsize=(20, 20),
                #                 fontsize=20,
                #                 save_path=f"results/examples/batch_{idx}.png")                  
    
        mean_train_loss = np.mean(tot_train_loss)
        mean_val_loss = np.mean(tot_val_loss)

        print(
            f"[info] Train loss on {epoch+1} epoch: BCE={mean_train_loss}, validation: BCE={mean_val_loss}"
        )

        with open(save_loss_dir, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_train_loss, mean_val_loss])

        # Plotting loss
        loss_plot(
            path2logs=save_loss_dir, savepath=save_plot_dir, logscale=False
        )

        # save model
        if (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    opt.save_model_path,
                    f"checkpoints/{opt.model_name}_{epoch}.pth",
                ),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--test_ratio", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--regularization", default=0, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgs_per_batch", default=2, type=int)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument("--model_name", default="UNet")
    parser.add_argument("--save_model_path", default="results")
    parser.add_argument("--load_model_path", default=None)
    parser.add_argument("--seed", default=42, type=int)
    opt = parser.parse_args()

    # NN model
    model = UNetSameSize(n_channels=3*opt.imgs_per_batch, n_classes=3)
    # model = nn.DataParallel(model)

    print("[Info]: Finish creating model!", flush=True)

    if opt.load_model_path is not None:
        pth = torch.load(opt.load_model_path)
        model.load_state_dict(pth)
        print("[Info]: Finish load pretrained weight!", flush=True)

    auto_create_path(opt.save_model_path)
    train(opt, model)
    # finding_pattern(opt, data, blending_model)
