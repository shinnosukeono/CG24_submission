import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import HourGlass, UNet  # noqa: F401
from PIL import Image
from torch import nn, optim
from tqdm import tqdm
from utils import create_strided_patches, downsample, psnr, sample_synthetic_image, seed_everything, timer, upsample


def bilateral_filter(
    image: np.ndarray,  # [H, W, C]
    sigma_spatial: float = 1.0,
    sigma_pixel: float = 0.2,
    filter_size: int = 3,
) -> np.ndarray:
    filter_half = filter_size // 2

    # spatial gaussian
    x, y = np.mgrid[-filter_half : filter_half + 1, -filter_half : filter_half + 1]
    gaussian_spatial = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))  # [filter_h, filter_w]

    # pixel gaussian
    patches = create_strided_patches(image, kernel_size=filter_size)  # [H, W, filter_h, filter_w, C]
    centers = np.tile(
        image[:, :, np.newaxis, np.newaxis, :],
        (1, 1, filter_size, filter_size, 1),
    )  # [H, W, filter_h, filter_w, C]
    gaussian_pixel = np.exp(
        -(np.linalg.norm(centers - patches, axis=-1) ** 2) / (2 * sigma_pixel**2)
    )  # [H, W, filter_h, filter_w]

    # weights
    w = gaussian_spatial[np.newaxis, np.newaxis, :, :] * gaussian_pixel  # [H, W, filter_h, filter_w]

    return np.einsum("ijklc,ijklc->ijc", w[:, :, :, :, np.newaxis], patches) / np.sum(w, axis=(2, 3))[:, :, np.newaxis]


def joint_bilateral_filter(
    image: np.ndarray,
    guidance: np.ndarray,
    sigma_spatial: float = 1.0,
    sigma_pixel: float = 0.2,
    filter_size: int = 3,
) -> np.ndarray:
    filter_half = filter_size // 2

    # spatial gaussian
    x, y = np.mgrid[-filter_half : filter_half + 1, -filter_half : filter_half + 1]
    gaussian_spatial = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))  # [filter_h, filter_w]

    # pixel gaussian
    patches_image = create_strided_patches(image, kernel_size=filter_size)  # [H, W, filter_h, filter_w, C]
    patches_guidance = create_strided_patches(guidance, kernel_size=filter_size)  # [H, W, filter_h, filter_w, C]
    centers_guidance = np.tile(
        guidance[:, :, np.newaxis, np.newaxis, :],
        (1, 1, filter_size, filter_size, 1),
    )
    gaussian_pixel = np.exp(
        -(np.linalg.norm(centers_guidance - patches_guidance, axis=-1) ** 2) / (2 * sigma_pixel**2)
    )  # [H, W, filter_h, filter_w]

    w = gaussian_spatial[np.newaxis, np.newaxis, :, :] * gaussian_pixel  # [H, W, filter_h, filter_w]

    return np.einsum("ijklc,ijklc->ijc", w[:, :, :, :, np.newaxis], patches_image) / np.sum(w, axis=(2, 3))[:, :, np.newaxis]


def pyramid_texture_filter(
    image: np.ndarray,
    depth: int = 11,
    scale: float = 0.8,
    sigma_spatial: float = 5.0,
    sigma_pixel: float = 0.07,
    filter_size: int = 3,
) -> np.ndarray:
    H, W, _ = image.shape

    depth = min(depth, int(np.emath.logn(1 / scale, min(H, W))))

    # pyramid
    Gs: list[np.ndarray] = []
    Ls: list[np.ndarray] = []

    image_down = image
    Gs.append(image)
    for _ in range(1, depth):
        image_prev = image_down
        down_output_size = int(image_prev.shape[0] * scale), int(image_prev.shape[1] * scale)
        up_output_size = image_prev.shape[0], image_prev.shape[1]
        image_down = downsample(image_down, filter_size, sigma_spatial, down_output_size)
        Gs.append(image_down)
        Ls.append(image_prev - upsample(image_down, filter_size, sigma_spatial, up_output_size))

    # PSU
    outputs = Gs[-1]
    for i in range(depth - 1, 0, -1):
        sigma_spatial_adaptive = sigma_spatial * pow(scale, i - 1)
        filter_size_upsample_adaptive = max(3, (lambda x: x if x % 2 == 1 else x + 1)(round(sigma_spatial_adaptive)))
        filter_size_adaptive = max(3, (lambda x: x if x % 2 == 1 else x + 1)(round(4 * sigma_spatial_adaptive)))
        up_output_size = Gs[i - 1].shape[0], Gs[i - 1].shape[1]
        jbf_upsampled = joint_bilateral_filter(
            upsample(outputs, filter_size, sigma_spatial, up_output_size),
            Gs[i - 1],
            sigma_spatial_adaptive,
            sigma_pixel,
            filter_size_upsample_adaptive,
        )
        outputs = joint_bilateral_filter(
            jbf_upsampled + Ls[i - 1],
            jbf_upsampled,
            sigma_spatial_adaptive,
            sigma_pixel,
            filter_size_adaptive,
        )

    return outputs


def guided_filter(
    image: np.ndarray,
    guidance: np.ndarray,
    filter_size: int = 3,
    epsilon: float = 1e-5,
):
    patches_image = create_strided_patches(image, filter_size)
    patches_guidance = create_strided_patches(guidance, filter_size)
    mus_image = patches_image.mean(axis=(2, 3))
    mus_guidance = patches_guidance.mean(axis=(2, 3))
    vars_guidance = patches_guidance.var(axis=(2, 3))

    a = np.mean(
        (patches_image * patches_guidance - (mus_guidance * mus_image)[:, :, np.newaxis, np.newaxis, :]),
        axis=(2, 3),
    ) / (
        vars_guidance + epsilon
    )  # [H, W, C]
    b = mus_image - a * mus_guidance  # [H, W, C]
    a_mean = np.mean(
        create_strided_patches(a, filter_size, mode="reflect"),
        axis=(2, 3),
    )
    b_mean = np.mean(
        create_strided_patches(b, filter_size, mode="reflect"),
        axis=(2, 3),
    )

    return a_mean * guidance + b_mean


def deep_image_prior(
    image: np.ndarray,
    n_epochs: int = 3000,
    lr: float = 0.01,
    backbone: nn.Module = HourGlass,
) -> list[np.ndarray]:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"{device=}")

    model = backbone()
    print(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    outputs = []
    model.to(device=device)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)  # expands to RGB
    image_true = (
        torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
    )  # [H, W, C] -> [B, C, H, W]
    x = (
        torch.rand(
            (1, 32, image.shape[0], image.shape[1]),
            dtype=torch.float32,
            device=device,
        )
        / 10
    )
    x_saved = x.detach().clone()
    noise = x.detach().clone()
    for epoch in tqdm(range(n_epochs)):
        optimizer.zero_grad()
        input = x_saved + (noise.normal_() / 30)
        output: torch.Tensor = model(input)
        loss = criterion(output, image_true)
        losses.append(loss)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            # print(f"Epoch: {epoch}, Loss: {loss.item(): .4f}")
            output = output.squeeze(0).permute(1, 2, 0).to(device="cpu").detach().numpy()
            outputs.append(output)

    return outputs[-1]


def eval(
    noise_image: np.ndarray,
    true_image: np.ndarray,
    denoiser: callable,
    guidance_image: np.ndarray | None = None,
    backbone: nn.Module = HourGlass,
) -> np.ndarray:
    if denoiser.__name__ == "guided_filter" or denoiser.__name__ == "joint_bilateral_filter":
        with timer(denoiser.__name__):
            filtered_image = denoiser(noise_image, guidance_image)
    elif denoiser.__name__ == "deep_image_prior":
        with timer(denoiser.__name__):
            filtered_image = denoiser(noise_image, backbone=backbone)
    else:
        with timer(denoiser.__name__):
            filtered_image = denoiser(noise_image)

    print(f"PSNR = {psnr(filtered_image, true_image)}")

    return filtered_image


if __name__ == "__main__":
    seed_everything(seed=42, gpu=True)
    if len(sys.argv) > 1:
        true_image = np.asarray(Image.open(sys.argv[1]).resize((256, 256))) / 255
        noise_image = np.clip(true_image + np.random.normal(scale=25 / 255, size=true_image.shape), 0, 1).astype(np.float32)
    else:
        true_image, noise_image = sample_synthetic_image(size=256)
        true_image = true_image[:, :, np.newaxis]
        noise_image = noise_image[:, :, np.newaxis]

    bilateral_result = eval(noise_image, true_image, bilateral_filter)
    ptf_result = eval(noise_image, true_image, pyramid_texture_filter)
    guidance_image, _ = sample_synthetic_image(size=256)
    guidance_image = guidance_image[:, :, np.newaxis]
    # jbf_result = eval(noise_image, true_image, joint_bilateral_filter, guidance_image)
    # guided_result = eval(noise_image, true_image, guided_filter, guidance_image)
    # dip_result = eval(noise_image, true_image, deep_image_prior)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(true_image)
    axes[0].set(title="Ground Truth")
    axes[1].imshow(np.clip(noise_image, 0, 1))
    axes[1].set(title="Input")
    axes[2].imshow(np.clip(bilateral_result, 0, 1, dtype=np.float32))
    axes[2].set(title="Bilateral Filter")
    axes[3].imshow(np.clip(ptf_result, 0, 1, dtype=np.float32))
    axes[3].set(title="Pyramid Texture Filter")

    plt.show()
    fig.savefig("result.png")
