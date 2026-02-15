import os
import torch
import numpy as np
import cv2
import json
from scipy.optimize import curve_fit
from scene.NVDIFFREC import util
import matplotlib.pyplot as plt

def gamma_tonemap(color, gamma=2.2):
    """Apply gamma tonemapping to an HDR color tensor.

    Args:
        color (torch.Tensor): Input HDR color tensor in the range [0,1].
        gamma (float): Gamma correction value (default 2.2 for sRGB).

    Returns:
        torch.Tensor: Tonemapped color tensor.
    """
    if isinstance(color, torch.Tensor):
        return torch.clamp(color ** (1.0 / gamma), 0, 1)
    elif isinstance(color, np.ndarray):
        return np.clip(color ** (1.0 / gamma), 0, 1)
    else:
        raise RuntimeWarning(f"gamma_tonemap is not defined for type {type(color)}")

def aces_tonemap(x):
    a,b,c,d,e = 2.51, 0.03, 2.43, 0.59, 0.14
    return torch.clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0)

# Example tonemapping function (e.g., a simple gamma correction)
def gamma_func(x, a, gamma, b):
    # a scales the input, gamma adjusts the non-linearity, and b is an offset
    return a * (x ** gamma) + b


def estimate_channel_tonemap(channel_A, channel_B):
    """ Function to extract channel data and perform curve fitting """
    # Flatten the channel arrays
    x = channel_A.flatten().astype(np.float64)
    y = channel_B.flatten().astype(np.float64)

    # Normalize using a robust maximum (e.g., the 99.9th percentile)
    max_x = np.percentile(x, 99.9)
    max_y = np.percentile(y, 99.9)

    # Clip and normalize the values to [0, 1]
    x_norm = np.clip(x / max_x, 0, 1)
    y_norm = np.clip(y / max_y, 0, 1)

    # Filter out outliers by ignoring values above a chosen threshold (e.g., 99th percentile)
    x_thresh = np.percentile(x_norm, 99)
    y_thresh = np.percentile(y_norm, 99)

    # Filter out outliers by ignoring values above a chosen threshold (e.g., 99th percentile)
    x_thresh = np.percentile(x_norm, 99)
    y_thresh = np.percentile(y_norm, 99)
    valid = (x_norm <= x_thresh) & (y_norm <= y_thresh)
    x_filtered = x_norm[valid]
    y_filtered = y_norm[valid]

    # Fit the tonemapping function to the filtered data
    popt, _ = curve_fit(gamma_func, x_filtered, y_filtered, p0=[1, 1, 0])

    return popt


def estimate_tonemap(a, b):
    """ Estimate tonemap for each channel
    Args:
        a (str or np.ndarray): Path to image A or the image itself
        b (str or np.ndarray): Path to image B or the image itself
    """
    def path_or_image(x):
        if isinstance(x, str):
            return util.load_image(x)
        elif isinstance(a, np.ndarray):
            return x
        else:
            raise ValueError("Invalid input type for image A: ", type(a))

    image_A = path_or_image(a)
    image_B = path_or_image(b)

    if image_A.shape != image_B.shape:
        image_B = cv2.resize(image_B, (image_A.shape[1], image_A.shape[0]), interpolation=cv2.INTER_CUBIC)

    popt = {}
    for i, channel in enumerate(['red', 'green', 'blue']):
        channel_A = image_A[:, :, i]
        channel_B = image_B[:, :, i]
        popt[channel] = list(estimate_channel_tonemap(channel_B, channel_A))

    tonemapped_B = apply_tonemapping(image_B, popt)
    return popt, tonemapped_B


def apply_tonemapping(envmap, params, robust_percentile=99.9):

    if not isinstance(envmap, np.ndarray) and isinstance(envmap, str):
        envmap = util.load_image(envmap) # Load the image if it's a path

    # Apply the tonemapping function to each channel
    envmap_tonemapped = np.zeros_like(envmap, dtype=np.float32)

    # Function to apply the tonemapping function on a single channel.
    def apply_tonemap_channel(channel, params, robust_percentile=99.9):
        # Compute a robust maximum to use for normalization (HDR images have extended range)
        max_val = np.percentile(channel, robust_percentile)
        # Normalize the channel into [0, 1]
        channel_norm = np.clip(channel / max_val, 0, 1)
        # Apply the learned tonemap function (element-wise)
        channel_tonemapped = gamma_func(channel_norm, *params)
        # Clip result to [0,1] (if desired)
        channel_tonemapped = np.clip(channel_tonemapped, 0, 1)
        # Optionally, scale back to the original dynamic range if needed:
        # channel_tonemapped *= max_val
        return channel_tonemapped

    for i in range(3):
        envmap_tonemapped[:, :, i] = apply_tonemap_channel(envmap[:, :, i], params[['red', 'green', 'blue'][i]], robust_percentile)

    return envmap_tonemapped


def save_tonemap_params(tonemap_params, path):
    with open(os.path.join(path, "tonemap_params.json"), "w") as f:
        json.dump(tonemap_params, f, indent=4)


def load_tonemap_params(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Tonemapping params file not found at {path}")
    with open(path, "r") as f:
        loaded_params = json.load(f)
        print("Estimated tonemapping params: ", loaded_params)

    return loaded_params

################################################################################
""" Borrowed from NMF codebase https://github.com/half-potato/nmf/blob/main/modules/tonemap.py """
class Tonemap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def white(self):
        self.tonemap(torch.tensor(1)).item()


class Filmic(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to SRGB
        # img from 0 to 1
        limit = 0.0031308
        out = torch.where(img > limit, 1.055 * (img.clip(min=limit) ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        # mask = img > limit
        # out = torch.zeros_like(img)
        # out[mask] = 1.055 * (img[mask] ** (1.0 / 2.4)) - 0.055
        # out[~mask] = 12.92 * img[~mask]
        if not noclip:
            out = out.clip(0, 1)
        return out

    def inverse(self, img):
        # SRGB to linear
        # img from 0 to 1
        limit = 0.04045
        return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)


class SRGBTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to SRGB
        # img from 0 to 1
        limit = 0.0031308
        out = torch.where(img > limit, 1.055 * (img.clip(min=limit) ** (1.0 / 2.4)) - 0.055, 12.92 * img)
        # mask = img > limit
        # out = torch.zeros_like(img)
        # out[mask] = 1.055 * (img[mask] ** (1.0 / 2.4)) - 0.055
        # out[~mask] = 12.92 * img[~mask]
        if not noclip:
            out = out.clip(0, 1)
        return out

    def inverse(self, img):
        # SRGB to linear
        # img from 0 to 1
        limit = 0.04045
        return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)


class HDRTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to HDR
        # reinhard hdr mapping + gamma correction
        out = (img / (img.clip(min=0)+1))**(1/2.2)
        if not noclip:
            out = out.clip(0, 1)
        return out

    def inverse(self, img):
        # HDR to linear
        img = img ** 2.2
        return - img / (img-1)


class LinearTonemap(Tonemap):
    def __init__(self):
        super().__init__()

    def forward(self, img, noclip=False):
        # linear to HDR
        if not noclip:
            img = img.clip(0, 1)
        return img

    def inverse(self, img):
        # HDR to linear
        return img

