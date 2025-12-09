import os
import os.path as osp
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

from utils.noise import jahne_noise
from utils.stuff import expand_zeros
from utils.conv import fft_conv

class DenoiseDataset(Dataset):
    """Datset for Denlising challenge"""

    def __init__(
        self,
        image_dir: str,
        psf_dir: str,
        batchsize: int = 2,
        transform: transforms = None,
    ):
        """Class constructor"""

        # Load transforms
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        self.batchsize = batchsize

        # Make paths
        self.image_paths = []
        self.psf_paths = []

        for img_path in sorted(os.listdir(image_dir)):
            if img_path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                full_path = osp.abspath(
                    osp.join(image_dir, img_path)
                )
                self.image_paths.append(full_path)

        for psf_path in sorted(os.listdir(psf_dir)):
            if psf_path.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                full_path = osp.abspath(
                    osp.join(psf_dir, psf_path)
                )
                self.psf_paths.append(full_path)

        np.random.seed(42)
    
        self.pairs = []
        for im_index in range(len(self.image_paths)):
            # Choosing random indexes from images
            random_indices = np.random.choice(len(self.psf_paths), 
                                              size=batchsize, 
                                              replace=False)
            
            self.pairs.append((im_index, random_indices))

    def __len__(self):
        return len(self.pairs)
    
    def add_psf(self, image, psf=None):
        if psf is None:
            return image
        return fft_conv(image, psf)

    def add_noise_and_blur(self, image, psfs):
        std = 1e-2
        base_lam = 1e-2

        result = jahne_noise(self.add_psf(image, psfs[0]), base_lam, std)
        for i in range(1, self.batchsize):
            image_noise = jahne_noise(self.add_psf(image, psfs[i]), base_lam / (i + 1), std)
            result = torch.concat([result, image_noise], dim=0)

        return result

    def __getitem__(self, idx):
        im_idx, psf_idx = self.pairs[idx]

        # Reading image
        image = cv2.imread(self.image_paths[im_idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reading psf
        psfs = []
        for idx in psf_idx:
            psf = cv2.imread(self.psf_paths[idx], cv2.IMREAD_GRAYSCALE)
            psf = expand_zeros(psf, image.shape[:2])
            psf = psf / psf.sum() # norm by sum
            psf = torch.from_numpy(psf).to(torch.float32)
            psfs.append(torch.fft.fftshift(psf))

        image = self.transform(image)
        images = self.add_noise_and_blur(image, psfs)

        return images, image



def getDenoiseLoader(
    image_dir: str,
    psf_dir: str,
    imgs_per_batch: int = 2,
    batchsize: int = 1,
    shuffle: bool = True,
    val_ratio: float = 0,
    test_ratio: float = 0,
    seed: int = 12345
):
    transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop((256, 256)),
                                        transforms.ToTensor()
                                   ])

    dataset = DenoiseDataset(image_dir, psf_dir, imgs_per_batch)

    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=shuffle, generator=generator
    )

    val_dataloader = None
    test_dataloader = None

    if val_size != 0:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batchsize, shuffle=shuffle, generator=generator
        )

    if test_size != 0:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batchsize, shuffle=shuffle, generator=generator
        )

    return train_dataloader, val_dataloader, test_dataloader

