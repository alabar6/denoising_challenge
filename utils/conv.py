import torch

def fft_conv(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    FFT base method for fast convolution. Input types are expected to be
    `float32` and size must be equal
    """
    assert image.dtype == torch.float32, image.dtype
    assert kernel.dtype == torch.float32, kernel.dtype
    assert (
        image.shape[-2:] == kernel.shape[-2:]
    ), f"Expected equal shapes, got: image={image.shape[-2:]}, kernel={kernel.shape[-2:]}"

    return torch.real(
        torch.fft.ifft2(torch.fft.fft2(image) * torch.fft.fft2(kernel))
    )