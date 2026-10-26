"""
SPR 2017 paper-model v2.

This standalone implementation follows the paper pipeline consistently:

    virtual RGB image -> adaptive main/NN filters -> RGBG Diamond samples
    RGBG samples -> bilinear virtual RGB reconstruction

The display grid is k times lower than the virtual image grid (k=2 by default).
The R/B geometry is the standard checkerboard Diamond approximation because the
paper does not publish panel-specific physical coordinates or all filter values.
Unlike the earlier versions, filter derivation and runtime reconstruction use the
same bilinear reconstruction matrices R_c; no Delaunay interpolation is used.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass(frozen=True)
class LayoutConfig:
    k: int = 2
    rb_x_offset: int = 1
    rb_y_offset: int = 0


@dataclass(frozen=True)
class FilterConfig:
    fmain_g: int = 9
    fmain_r: int = 13
    fmain_b: int = 13
    fnn_g: int = 3
    fnn_r: int = 7
    fnn_b: int = 7


def srgb_to_linear_u8(image: np.ndarray, gamma: float) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    return np.power(np.clip(values, 0.0, 1.0), gamma).astype(np.float32)


def linear_to_srgb_u8(image: np.ndarray, gamma: float) -> np.ndarray:
    values = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
    return np.clip(np.round(values * 255.0), 0, 255).astype(np.uint8)


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    total = float(np.sum(kernel))
    if abs(total) < 1e-12:
        return kernel.astype(np.float32)
    return (kernel / total).astype(np.float32)


def center_crop(kernel: np.ndarray, size: int) -> np.ndarray:
    center_y, center_x = (value // 2 for value in kernel.shape)
    radius = size // 2
    return kernel[center_y - radius:center_y + radius + 1,
                  center_x - radius:center_x + radius + 1].copy()


def crop_kernel(kernel: np.ndarray, center_y: int, center_x: int, size: int) -> np.ndarray:
    radius = size // 2
    padding = radius + 2
    padded = np.pad(kernel, ((padding, padding), (padding, padding)), mode="constant")
    center_y += padding
    center_x += padding
    return padded[center_y - radius:center_y + radius + 1,
                  center_x - radius:center_x + radius + 1].astype(np.float32)


def pad_kernel(kernel: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float32)
    row_offset = (size - kernel.shape[0]) // 2
    column_offset = (size - kernel.shape[1]) // 2
    result[row_offset:row_offset + kernel.shape[0], column_offset:column_offset + kernel.shape[1]] = kernel
    return result


def make_nonnegative_filter(main: np.ndarray, size: int) -> np.ndarray:
    result = center_crop(main, size).astype(np.float32)
    result[result < 0] = 0.0
    return normalize_kernel(result)


def alpha_from_filter(kernel: np.ndarray) -> float:
    return float(-np.sum(kernel[kernel < 0]))


def diamond_masks(row_count: int, column_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.indices((row_count, column_count))
    green = np.ones((row_count, column_count), dtype=bool)
    red = ((rows + columns) % 2 == 0)
    blue = ~red
    return red, green, blue


def subpixel_points(row_count: int, column_count: int, channel: str, config: LayoutConfig) -> Tuple[np.ndarray, np.ndarray]:
    red, green, blue = diamond_masks(row_count, column_count)
    mask = {"r": red, "g": green, "b": blue}[channel]
    rows, columns = np.where(mask)
    points_x = config.k * columns
    points_y = config.k * rows
    if channel in ("r", "b"):
        points_x = np.minimum(points_x + config.rb_x_offset, config.k * column_count - 1)
        points_y = np.minimum(points_y + config.rb_y_offset, config.k * row_count - 1)
    points = np.stack((points_x, points_y), axis=1).astype(np.int32)
    return points, mask


def green_bilinear_weight(dx: int, dy: int, k: int) -> float:
    weight_x = 1.0 - abs(dx) / float(k)
    weight_y = 1.0 - abs(dy) / float(k)
    if weight_x <= 0.0 or weight_y <= 0.0:
        return 0.0
    return weight_x * weight_y


def diamond_bilinear_weight(dx: int, dy: int, k: int) -> float:
    coordinate_u = (dx + dy) / float(2 * k)
    coordinate_v = (dx - dy) / float(2 * k)
    weight_u = 1.0 - abs(coordinate_u)
    weight_v = 1.0 - abs(coordinate_v)
    if weight_u <= 0.0 or weight_v <= 0.0:
        return 0.0
    return weight_u * weight_v


def bilinear_offsets(channel: str, k: int) -> Iterable[Tuple[int, int, float]]:
    radius = k - 1 if channel == "g" else 2 * k - 1
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = (green_bilinear_weight(dx, dy, k) if channel == "g"
                      else diamond_bilinear_weight(dx, dy, k))
            if weight > 1e-12:
                yield dy, dx, weight


def build_bilinear_matrix(row_count: int, column_count: int, channel: str, config: LayoutConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    virtual_row_count = config.k * row_count
    virtual_column_count = config.k * column_count
    points, mask = subpixel_points(row_count, column_count, channel, config)
    matrix = np.zeros((virtual_row_count * virtual_column_count, points.shape[0]), dtype=np.float32)
    offsets = tuple(bilinear_offsets(channel, config.k))
    for column, (point_x, point_y) in enumerate(points):
        for dy, dx, weight in offsets:
            y = int(point_y + dy)
            x = int(point_x + dx)
            if 0 <= y < virtual_row_count and 0 <= x < virtual_column_count:
                matrix[y * virtual_column_count + x, column] += weight
    return matrix, points, mask


@lru_cache(maxsize=12)
def cached_bilinear_matrix(row_count: int, column_count: int, channel: str, k: int, offset_x: int, offset_y: int):
    return build_bilinear_matrix(row_count, column_count, channel, LayoutConfig(k, offset_x, offset_y))


def derive_main_filter(channel: str, size: int, design_size: int, config: LayoutConfig, rcond: float) -> np.ndarray:
    matrix, points, _ = cached_bilinear_matrix(
        design_size, design_size, channel, config.k, config.rb_x_offset, config.rb_y_offset
    )
    inverse = np.linalg.pinv(matrix.astype(np.float64), rcond=rcond)
    center = np.array([(config.k * design_size - 1) / 2.0] * 2, dtype=np.float32)
    point_index = int(np.argmin(np.sum((points.astype(np.float32) - center) ** 2, axis=1)))
    full_filter = inverse[point_index].reshape(config.k * design_size, config.k * design_size)
    point_x, point_y = points[point_index]
    return normalize_kernel(crop_kernel(full_filter, int(point_y), int(point_x), size))


def expected_filter_shapes(config: FilterConfig) -> Dict[str, Tuple[int, int]]:
    return {
        "main_g": (config.fmain_g, config.fmain_g),
        "main_r": (config.fmain_r, config.fmain_r),
        "main_b": (config.fmain_b, config.fmain_b),
        "nn_g": (config.fnn_g, config.fnn_g),
        "nn_r": (config.fnn_r, config.fnn_r),
        "nn_b": (config.fnn_b, config.fnn_b),
    }


def prepare_filters(cache_path: str, config: LayoutConfig, filter_config: FilterConfig,
                    design_size: int, rcond: float, force_derive: bool) -> Dict[str, np.ndarray]:
    shapes = expected_filter_shapes(filter_config)
    if os.path.exists(cache_path) and not force_derive:
        with np.load(cache_path) as data:
            filters = {name: data[name].astype(np.float32) for name in data.files}
        if all(filters.get(name, np.empty(0)).shape == shape for name, shape in shapes.items()):
            return filters
        print("Filter cache shape mismatch; deriving the v2 paper-model filters.")

    print("Deriving v2 filters from the common bilinear R_c model...")
    main_g = derive_main_filter("g", filter_config.fmain_g, design_size, config, rcond)
    main_r = derive_main_filter("r", filter_config.fmain_r, design_size, config, rcond)
    main_b = derive_main_filter("b", filter_config.fmain_b, design_size, config, rcond)
    filters = {
        "main_g": main_g,
        "main_r": main_r,
        "main_b": main_b,
        "nn_g": make_nonnegative_filter(main_g, filter_config.fnn_g),
        "nn_r": make_nonnegative_filter(main_r, filter_config.fnn_r),
        "nn_b": make_nonnegative_filter(main_b, filter_config.fnn_b),
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(cache_path, **filters)
    return filters


def compute_beta(patch: np.ndarray, alpha: float, beta_scale: float, beta_exponent: float) -> float:
    if alpha <= 1e-12:
        return 0.0
    maximum = float(np.max(patch))
    minimum = float(np.min(patch))
    beta_over = 0.0
    beta_under = 0.0
    overflow_limit = 1.0 - (alpha / (1.0 + alpha)) * (1.0 - minimum)
    if maximum > overflow_limit:
        beta_over = 1.0 if maximum >= 1.0 - 1e-12 else 1.0 - alpha * (1.0 - minimum) / ((1.0 + alpha) * (1.0 - maximum))
    underflow_limit = alpha / (1.0 + alpha) * maximum
    if minimum < underflow_limit:
        beta_under = 0.0 if maximum <= 1e-12 else 1.0 - (1.0 + alpha) * minimum / (alpha * maximum)
    base_beta = max(beta_over, beta_under)
    return float(np.clip(beta_scale * (base_beta ** beta_exponent), 0.0, 1.0))


def filtered_sample(image: np.ndarray, y: int, x: int, main: np.ndarray, nn: np.ndarray,
                    alpha: float, beta_scale: float, beta_exponent: float,
                    padding: int) -> Tuple[float, float]:
    size = main.shape[0]
    center_y = y + padding
    center_x = x + padding
    patch = image[center_y - size // 2:center_y + size // 2 + 1,
                  center_x - size // 2:center_x + size // 2 + 1]
    beta = compute_beta(patch, alpha, beta_scale, beta_exponent)
    fused = beta * pad_kernel(nn, size) + (1.0 - beta) * main
    return float(np.clip(np.sum(patch * fused), 0.0, 1.0)), beta


def render_rgbg(virtual_rgb: np.ndarray, filters: Dict[str, np.ndarray], config: LayoutConfig,
                gamma: float, beta_scale: float, beta_exponent: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    virtual_height, virtual_width = virtual_rgb.shape[:2]
    if virtual_height % config.k or virtual_width % config.k:
        raise ValueError("Input image dimensions must be divisible by k=2.")
    height, width = virtual_height // config.k, virtual_width // config.k
    red_mask, green_mask, blue_mask = diamond_masks(height, width)
    planes = [np.zeros((height, width), dtype=np.float32) for _ in range(3)]
    beta_maps = [np.zeros((height, width), dtype=np.float32) for _ in range(3)]
    maximum_filter_size = max(filters[name].shape[0] for name in ("main_g", "main_r", "main_b"))
    padding = maximum_filter_size // 2 + 2
    padded_channels = [np.pad(virtual_rgb[..., index], ((padding, padding),) * 2, mode="reflect") for index in range(3)]
    masks = (red_mask, green_mask, blue_mask)
    channels = (("r", 0, red_mask), ("g", 1, green_mask), ("b", 2, blue_mask))
    for channel, index, mask in channels:
        main = filters[f"main_{channel}"]
        nn = filters[f"nn_{channel}"]
        alpha = alpha_from_filter(main)
        for y, x in zip(*np.where(mask)):
            virtual_y = config.k * int(y) + (config.rb_y_offset if channel != "g" else 0)
            virtual_x = config.k * int(x) + (config.rb_x_offset if channel != "g" else 0)
            value, beta = filtered_sample(
                padded_channels[index], virtual_y, virtual_x, main, nn, alpha,
                beta_scale, beta_exponent, padding
            )
            planes[index][y, x] = value
            beta_maps[index][y, x] = beta
    return planes[0], planes[1], planes[2], masks, {"r": beta_maps[0], "g": beta_maps[1], "b": beta_maps[2]}


def reconstruct_virtual_linear(planes: Tuple[np.ndarray, np.ndarray, np.ndarray], masks: Tuple[np.ndarray, np.ndarray, np.ndarray], config: LayoutConfig) -> np.ndarray:
    row_count, column_count = planes[1].shape
    virtual_row_count, virtual_column_count = config.k * row_count, config.k * column_count
    reconstructed = []
    for plane, channel, mask in zip(planes, ("r", "g", "b"), masks):
        points, _ = subpixel_points(row_count, column_count, channel, config)
        values = plane[mask]
        output = np.zeros((virtual_row_count, virtual_column_count), dtype=np.float32)
        weights = np.zeros_like(output)
        for value, (point_x, point_y) in zip(values, points):
            for dy, dx, weight in bilinear_offsets(channel, config.k):
                y = int(point_y + dy)
                x = int(point_x + dx)
                if 0 <= y < virtual_row_count and 0 <= x < virtual_column_count:
                    output[y, x] += float(value) * weight
                    weights[y, x] += weight
        covered = weights > 1e-12
        output[covered] /= weights[covered]
        if np.any(~covered):
            missing_y, missing_x = np.where(~covered)
            for y, x in zip(missing_y, missing_x):
                distance = (points[:, 1] - y) ** 2 + (points[:, 0] - x) ** 2
                output[y, x] = float(values[int(np.argmin(distance))])
        reconstructed.append(np.clip(output, 0.0, 1.0))
    return np.stack(reconstructed, axis=-1).astype(np.float32)


def rgb_to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    red, green, blue = [rgb[..., index].astype(np.float32) for index in range(3)]
    return (0.299 * red + 0.587 * green + 0.114 * blue,
            -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0,
            0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0)


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float32) - second.astype(np.float32)) ** 2))
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def metrics(reference: np.ndarray, result: np.ndarray) -> Dict[str, float]:
    _, cb_reference, cr_reference = rgb_to_ycbcr(reference)
    _, cb_result, cr_result = rgb_to_ycbcr(result)
    try:
        similarity = float(ssim(reference, result, channel_axis=-1, data_range=255))
    except TypeError:
        similarity = float(ssim(reference, result, multichannel=True, data_range=255))
    return {
        "MSE_RGB": float(np.mean((reference.astype(np.float32) - result.astype(np.float32)) ** 2)),
        "PSNR_RGB": psnr(reference, result),
        "SSIM_RGB": similarity,
        "PSNR_Cb": psnr(cb_reference, cb_result),
        "PSNR_Cr": psnr(cr_reference, cr_result),
    }


def process_image(path: str, output_directory: str, filters: Dict[str, np.ndarray], config: LayoutConfig,
                  gamma: float, beta_scale: float, beta_exponent: float) -> Dict[str, float]:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height = rgb.shape[0] - rgb.shape[0] % config.k
    width = rgb.shape[1] - rgb.shape[1] % config.k
    rgb = rgb[:height, :width]
    linear_rgb = srgb_to_linear_u8(rgb, gamma)
    red, green, blue, masks, beta_maps = render_rgbg(
        linear_rgb, filters, config, gamma, beta_scale, beta_exponent
    )
    virtual_linear = reconstruct_virtual_linear((red, green, blue), masks, config)
    virtual_rgb = linear_to_srgb_u8(virtual_linear, gamma)
    os.makedirs(output_directory, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(output_directory, f"{base}_v2_virtual.png"), cv2.cvtColor(virtual_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_directory, f"{base}_v2_R_sub.png"), linear_to_srgb_u8(red, gamma))
    cv2.imwrite(os.path.join(output_directory, f"{base}_v2_G_sub.png"), linear_to_srgb_u8(green, gamma))
    cv2.imwrite(os.path.join(output_directory, f"{base}_v2_B_sub.png"), linear_to_srgb_u8(blue, gamma))
    for channel, beta in beta_maps.items():
        cv2.imwrite(os.path.join(output_directory, f"{base}_v2_beta_{channel}.png"), np.clip(np.round(beta * 255.0), 0, 255).astype(np.uint8))
    return metrics(rgb, virtual_rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="SPR 2017 paper-model v2: common bilinear R_c pipeline.")
    parser.add_argument("images", nargs="+", help="Input virtual RGB image paths")
    parser.add_argument("--out", default="SPR2017_PaperV2_Result")
    parser.add_argument("--cache", default="spr2017_paper_v2_filters_cache.npz")
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--design-size", type=int, default=29)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument(
        "--beta-scale", type=float, default=1.0,
        help="Scale adaptive NN-filter weight beta; lower values preserve more sharpness."
    )
    parser.add_argument(
        "--beta-exponent", type=float, default=1.0,
        help="Shape beta blending; values above 1 reduce weak NN blending."
    )
    args = parser.parse_args()
    if args.beta_scale < 0.0 or args.beta_exponent <= 0.0:
        parser.error("--beta-scale must be nonnegative and --beta-exponent must be positive")
    config = LayoutConfig(k=2)
    filter_config = FilterConfig()
    filters = prepare_filters(args.cache, config, filter_config, args.design_size, args.rcond, args.derive)
    for name, kernel in filters.items():
        print(f"{name:7s}: shape={kernel.shape}, sum={np.sum(kernel):.6f}, alpha={alpha_from_filter(kernel):.6f}")
    rows = []
    for image_path in args.images:
        print(f"Processing {image_path}")
        result = process_image(
            image_path, args.out, filters, config, args.gamma,
            args.beta_scale, args.beta_exponent
        )
        result["image"] = os.path.basename(image_path)
        rows.append(result)
        print("  " + ", ".join(f"{key}={value:.4f}" for key, value in result.items() if key != "image"))
    if rows:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metrics_summary.csv"), "w", newline="", encoding="utf-8") as handle:
            fields = ["image", "MSE_RGB", "PSNR_RGB", "SSIM_RGB", "PSNR_Cb", "PSNR_Cr"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


if __name__ == "__main__":
    main()
