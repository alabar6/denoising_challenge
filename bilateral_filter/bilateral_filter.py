import os.path as osp
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import getDenoiseLoader

def bilateral_batch_opencv(noisy_batch: torch.Tensor, num_patches: int,
                           d=15, sigma_color=75, sigma_space=75):
    B, _, H, W = noisy_batch.shape
    denoised_patches = []

    for i in range(num_patches):
        patch = noisy_batch[:, i*3:(i+1)*3, :, :]
        filtered = []
        for b in range(B):
            img_np = (patch[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            filtered_np = cv2.bilateralFilter(img_np, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).float() / 255.0
            filtered.append(filtered_tensor.to(noisy_batch.device))
        denoised_patches.append(torch.stack(filtered))

    return torch.stack(denoised_patches).mean(dim=0).clamp(0, 1)

def bilateral_batch_custom(noisy_batch: torch.Tensor, num_patches: int,
                           kernel_size=15, sigma_space=15.0, sigma_color=75.0,
                           spatial_type: str = "gaussian",
                           range_type: str = "gaussian"):
    """
    [B, 3*P, H, W] → [B, 3, H, W]
    Работает на всех версиях PyTorch.
    """
    B, C_full, H, W = noisy_batch.shape
    assert C_full == 3 * num_patches, f"Expected 3*{num_patches} channels, got {C_full}"

    device = noisy_batch.device
    pad = kernel_size // 2
    N = kernel_size * kernel_size

    # [B, P, 3, H, W]
    patches = noisy_batch.view(B, num_patches, 3, H, W)

    # Сливаем B и P → [B*P, 3, H, W]
    patches_flat = patches.reshape(B * num_patches, 3, H, W)

    # Паддинг reflect (4D — работает)
    padded_flat = F.pad(patches_flat, (pad, pad, pad, pad), mode='reflect')  # [B*P, 3, H+2p, W+2p]

    # F.unfold для извлечения всех патчей kernel_size x kernel_size
    # output: [B*P, 3*kernel_size*kernel_size, L] где L = H*W
    unfolded_flat = F.unfold(padded_flat, kernel_size=(kernel_size, kernel_size),
                             padding=0, stride=1)  # [B*P, 3*N, H*W]

    # [B*P, 3, N, H*W]
    unfolded_flat = unfolded_flat.view(B * num_patches, 3, N, H * W)

    # Центр: [B*P, 3, 1, H*W]
    center_flat = patches_flat.view(B * num_patches, 3, 1, H * W)

    # Разница по каналам
    diff = unfolded_flat - center_flat  # [B*P, 3, N, H*W]
    diff_sq = (diff ** 2).sum(dim=1)     # [B*P, N, H*W]

    # Range kernel
    if range_type == "gaussian":
        range_weight = torch.exp(-diff_sq / (2.0 * sigma_color ** 2))
    elif range_type == "exp_l1":
        range_weight = torch.exp(-torch.sqrt(diff_sq + 1e-8) / sigma_color)
    elif range_type == "box":
        range_weight = (diff_sq <= (sigma_color ** 2)).float()
    else:
        raise ValueError("range_type: 'gaussian', 'exp_l1', 'box'")

    # Spatial kernel (фиксированная)
    r = torch.arange(-pad, pad + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing='ij')
    spatial_dist_sq = xx**2 + yy**2
    spatial_dist_sq = spatial_dist_sq.flatten()  # [N]

    if spatial_type == "gaussian":
        spatial_weight = torch.exp(-spatial_dist_sq / (2.0 * sigma_space ** 2))
    elif spatial_type == "exp_l1":
        spatial_weight = torch.exp(-torch.sqrt(spatial_dist_sq + 1e-8) / sigma_space)
    elif spatial_type == "box":
        spatial_weight = (spatial_dist_sq <= (pad ** 2)).float()
    else:
        raise ValueError("spatial_type: 'gaussian', 'exp_l1', 'box'")

    spatial_weight = spatial_weight.view(1, N, 1)  # [1, N, 1]

    # Общий вес
    weight = range_weight * spatial_weight  # [B*P, N, H*W]
    weight_sum = weight.sum(dim=1, keepdim=True)  # [B*P, 1, H*W]

    # Взвешенная сумма
    weighted_sum = (unfolded_flat * weight.unsqueeze(1)).sum(dim=2)  # [B*P, 3, H*W]

    # Нормализация
    filtered_flat = weighted_sum / (weight_sum + 1e-8)  # [B*P, 3, H*W]

    # Возвращаем форму [B*P, 3, H, W]
    filtered_flat = filtered_flat.view(B * num_patches, 3, H, W)

    # [B, P, 3, H, W]
    filtered_patches = filtered_flat.view(B, num_patches, 3, H, W)

    # Усреднение по патчам
    final = filtered_patches.mean(dim=1)  # [B, 3, H, W]
    return final.clamp(0, 1)


def bilateral_denoise_batch(
    noisy_batch: torch.Tensor,
    num_patches: int,
    filter_type: str = "custom",
    d: int = 15,
    kernel_size: int = 15,
    sigma_color: float = 75,
    sigma_space: float = 75,
    spatial_type: str = "gaussian",
    range_type: str = "gaussian",
):
    if filter_type == "opencv":
        return bilateral_batch_opencv(noisy_batch, num_patches, d=d,
                                      sigma_color=sigma_color, sigma_space=sigma_space)
    elif filter_type == "custom":
        return bilateral_batch_custom(noisy_batch, num_patches,
                                      kernel_size=kernel_size,
                                      sigma_color=sigma_color, sigma_space=sigma_space,
                                      spatial_type=spatial_type, range_type=range_type)
    else:
        raise ValueError("filter_type: 'opencv' или 'custom'")

def process_folder(
    input_dir: str,
    output_dir: str,
    num_patches: int = 8,
    batch_size: int = 1,
    filter_type: str = "custom",
    d: int = 15,
    kernel_size: int = 15,
    sigma_color: float = 0.08,   
    sigma_space: float = 15.0,
    spatial_type: str = "gaussian",
    range_type: str = "gaussian",
):
    os.makedirs(output_dir, exist_ok=True)

    dataloader, _, _ = getDenoiseLoader(
        image_dir=input_dir,
        psf_dir=None,
        imgs_per_batch=num_patches,
        batchsize=batch_size,
        shuffle=False,
        val_ratio=0,
        test_ratio=0,
    )

    to_pil = transforms.ToPILImage()

    print(f"Обрабатываем {len(dataloader)} изображений")
    print(f"Патчи: {num_patches} | Фильтр: {filter_type.upper()}")
    print(f"Spatial: {spatial_type} | Range: {range_type}")
    print(f"sigma_color={sigma_color} | sigma_space={sigma_space}")

    for idx, (noisy_batch, clean_gt) in enumerate(tqdm(dataloader, desc="Denoising")):
        denoised = bilateral_denoise_batch(
            noisy_batch,
            num_patches=num_patches,
            filter_type=filter_type,
            d=d,
            kernel_size=kernel_size,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
            spatial_type=spatial_type,
            range_type=range_type,
        )

        for b in range(denoised.shape[0]):
            dataset_idx = idx * batch_size + b
            orig_path = dataloader.dataset.dataset.image_paths[dataset_idx]
            filename = osp.basename(orig_path)

            pil_img = to_pil(denoised[b].cpu())
            save_path = osp.join(output_dir, filename)
            pil_img.save(save_path)

    print(f"Готово → {output_dir}")

if __name__ == "__main__":
    INPUT_DIR = "/Users/konstantinkornilov/Desktop/denoising_challenge/images"
    OUTPUT_DIR = "/Users/konstantinkornilov/Desktop/denoising_challenge/bilateral_denoised_custom"

    process_folder(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_patches=2,
        batch_size=16,
        filter_type="custom",
        kernel_size=15,
        sigma_color=0.08,
        sigma_space=8.0,
        spatial_type="gaussian",
        range_type="exp_l1",  
    )