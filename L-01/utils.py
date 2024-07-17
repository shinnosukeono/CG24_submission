import math
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import RegularGridInterpolator


def sample_synthetic_image(size: int = 256) -> tuple[np.ndarray]:
    true_image = np.concatenate((np.zeros((size // 2, size)), np.ones((size // 2, size))), axis=0)
    noise_image = np.clip(true_image + np.random.normal(0, 0.2, true_image.shape), 0.0, 1.0)
    return true_image, noise_image


def create_strided_patches(
    image: np.ndarray,
    kernel_size: int,
    mode: str = "edge",
) -> np.ndarray:
    H, W, C = image.shape
    padded_image = np.pad(
        image,
        pad_width=[(kernel_size // 2,), (kernel_size // 2,), (0,)],
        mode=mode,
    )
    stride_h, stride_w, stride_c = padded_image.strides
    output_shape = (H, W, kernel_size, kernel_size, C)
    output_strides = (stride_h, stride_w, stride_h, stride_w, stride_c)
    strided_image = as_strided(padded_image, shape=output_shape, strides=output_strides, writeable=False)
    return strided_image


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    kernel_range = np.arange(-size // 2 + 1, size // 2 + 1)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_filter(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    kernel = create_gaussian_kernel(kernel_size, sigma)
    patches = create_strided_patches(image, kernel_size, mode="symmetric")
    return np.sum(patches * kernel[np.newaxis, np.newaxis, :, :, np.newaxis], axis=(2, 3))


def downsample(image: np.ndarray, kernel_size: int, sigma: float, output_size: tuple[int, int]):
    H, W, C = image.shape
    filtered = gaussian_filter(image, kernel_size, sigma)

    new_H, new_W = output_size
    new_x = np.linspace(0, filtered.shape[0] - 1, new_H)
    new_y = np.linspace(0, filtered.shape[1] - 1, new_W)

    new_grid = np.meshgrid(new_x, new_y, indexing="ij")
    new_indices = np.vstack([new_grid[0].ravel(), new_grid[1].ravel()]).T

    new_image = np.empty((new_H, new_W, C))
    for i in range(C):
        interp = RegularGridInterpolator((np.arange(filtered.shape[0]), np.arange(filtered.shape[1])), filtered[:, :, i])
        new_image[:, :, i] = interp(new_indices).reshape(new_H, new_H)

    return new_image


def upsample(image: np.ndarray, kernel_size: int, sigma: float, output_size: tuple[int, int]):
    H, W, C = image.shape
    filtered = gaussian_filter(image, kernel_size, sigma)

    new_H, new_W = output_size
    new_x = np.linspace(0, filtered.shape[0] - 1, new_H)
    new_y = np.linspace(0, filtered.shape[1] - 1, new_W)

    new_grid = np.meshgrid(new_x, new_y, indexing="ij")
    new_indices = np.vstack([new_grid[0].ravel(), new_grid[1].ravel()]).T

    new_image = np.empty((new_H, new_W, C))
    for i in range(C):
        interp = RegularGridInterpolator((np.arange(filtered.shape[0]), np.arange(filtered.shape[1])), filtered[:, :, i])
        new_image[:, :, i] = interp(new_indices).reshape(new_H, new_H)

    return new_image


def psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    else:
        return 10 * np.log10(255**2 / mse)


def seed_everything(seed: int = 42, gpu=False):
    print(f"global seed set: {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if gpu:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


@contextmanager
def timer(name: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2.0**30
    print(f"<< {name} >> Start")
    yield

    m1 = p.memory_info()[0] / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)

    print(f"<< {name} >> {m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec")
