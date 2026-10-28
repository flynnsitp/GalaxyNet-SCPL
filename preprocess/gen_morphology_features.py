from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import gamma
from skimage.filters import threshold_otsu
from skimage.measure import label
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FEATURE_NAMES = [
    "C",
    "A",
    "S",
    "G",
    "M20",
    "ellipticity",
    "sersic_n",
    "half_light_radius",
    "B/T",
    "shape_asymmetry",
    "MID_M",
    "MID_I",
    "MID_D",
    "color_gradient",
]
INVALID_SENTINELS = {-99.0}
DEFAULT_FAILURE_PREVIEW_LIMIT = 20


@dataclass(frozen=True, slots=True)
class ValueDomain:
    lower: float | None = None
    upper: float | None = None


FEATURE_DOMAINS: dict[str, ValueDomain] = {
    "C": ValueDomain(lower=0.0),
    "A": ValueDomain(lower=0.0, upper=2.0),
    "S": ValueDomain(lower=0.0, upper=1.0),
    "G": ValueDomain(lower=0.0, upper=1.0),
    "M20": ValueDomain(upper=0.0),
    "ellipticity": ValueDomain(lower=0.0, upper=1.0),
    "sersic_n": ValueDomain(lower=0.3, upper=8.0),
    "half_light_radius": ValueDomain(lower=0.0),
    "B/T": ValueDomain(lower=0.0, upper=1.0),
    "shape_asymmetry": ValueDomain(lower=0.0, upper=1.0),
    "MID_M": ValueDomain(lower=0.0, upper=1.0),
    "MID_I": ValueDomain(lower=0.0, upper=1.0),
    "MID_D": ValueDomain(lower=0.0),
    "color_gradient": ValueDomain(lower=-2.0, upper=2.0),
}
AUXILIARY_BASE_DOMAINS: dict[str, ValueDomain] = {
    "mask_area": ValueDomain(lower=0.0),
    "centroid_y": ValueDomain(lower=0.0),
    "centroid_x": ValueDomain(lower=0.0),
    "r20": ValueDomain(lower=0.0),
    "r50": ValueDomain(lower=0.0),
    "r80": ValueDomain(lower=0.0),
    "major_sigma": ValueDomain(lower=0.0),
    "minor_sigma": ValueDomain(lower=0.0),
}
SANITIZE_COUNTS: Counter[str] = Counter()
SANITIZE_EXAMPLES: dict[str, list[str]] = defaultdict(list)
SANITIZE_EXAMPLE_LIMIT = DEFAULT_FAILURE_PREVIEW_LIMIT


@dataclass(slots=True)
class ImageContext:
    path: str
    rgb: np.ndarray
    gray: np.ndarray
    signal: np.ndarray
    mask: np.ndarray
    centroid_y: float
    centroid_x: float
    area: int
    ellipticity: float
    major_sigma: float
    minor_sigma: float


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute approximate galaxy morphology features for jpg/png images."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/flynn/work/datasets/astro/datasets/gz2_cls_seven_duplicated",
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file path. Default: saved under input_path as structural.json",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan sub-directories (default: enabled)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Disable recursive scan",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute records already present in the output JSON",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Checkpoint JSON after every N newly processed images",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=32,
        help="Minimum connected-component pixel area required for a valid galaxy mask",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation used when saving the JSON file",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Path to log file. Default: saved under input_path as morphology_features.log",
    )
    parser.add_argument(
        "--failure-preview-limit",
        type=int,
        default=DEFAULT_FAILURE_PREVIEW_LIMIT,
        help="Maximum number of failed image / feature examples printed per summary item.",
    )
    return parser


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("morphology_features")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def iter_image_files(input_path: str, recursive: bool = True) -> list[str]:
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {root}")
        return [str(root.resolve())]

    files: list[str] = []
    if recursive:
        walker: Iterable[Path] = root.rglob("*")
    else:
        walker = root.glob("*")

    for path in walker:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(str(path.resolve()))
    return sorted(files)


def load_existing_records(
    output_json: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    out_path = Path(output_json)
    if not out_path.exists():
        return [], {}

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Existing output JSON must contain a list of records.")

    records: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        image_path = os.path.abspath(str(item.get("path", "")))
        if not image_path:
            continue
        item["path"] = image_path
        records.append(item)
        mapping[image_path] = item
    return records, mapping


def save_records(output_json: str, records: list[dict[str, Any]], indent: int = 2) -> None:
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_sorted = sorted(
        records, key=lambda x: (str(x.get("path", "")), str(x.get("image_name", "")))
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records_sorted, f, ensure_ascii=False, indent=indent)


def load_image_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return rgb


def robust_background_subtract(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    border = max(2, min(h, w) // 16)
    border_pixels = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    bg_level = float(np.median(border_pixels))
    bg_sigma = float(1.4826 * np.median(np.abs(border_pixels - bg_level)) + 1e-6)
    signal = gray - bg_level
    signal[signal < 0.0] = 0.0
    signal[signal < 1.5 * bg_sigma] = 0.0
    return signal


def largest_component_mask(signal: np.ndarray, min_area: int = 32) -> np.ndarray:
    smooth = ndimage.gaussian_filter(signal, sigma=1.2)
    positive = smooth[smooth > 0]
    if positive.size == 0:
        raise ValueError("No positive signal after background subtraction")

    try:
        thresh = float(threshold_otsu(positive))
    except ValueError:
        thresh = float(np.percentile(positive, 75))
    thresh = max(thresh, float(np.percentile(positive, 55)))

    mask = smooth >= thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    labeled = label(mask)
    if labeled.max() == 0:
        raise ValueError("Segmentation failed: no connected component")

    component_ids, counts = np.unique(labeled[labeled > 0], return_counts=True)
    best_id = int(component_ids[np.argmax(counts)])
    best_mask = labeled == best_id
    best_mask = ndimage.binary_fill_holes(best_mask)
    area = int(best_mask.sum())
    if area < int(min_area):
        raise ValueError(f"Connected component too small: area={area}")
    return best_mask


def weighted_centroid(signal: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.indices(signal.shape, dtype=np.float64)
    weights = np.where(mask, np.clip(signal, 0.0, None), 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            raise ValueError("Mask is empty")
        return float(ys.mean()), float(xs.mean())
    cy = float((weights * yy).sum() / total)
    cx = float((weights * xx).sum() / total)
    return cy, cx


def second_moments(
    signal: np.ndarray, mask: np.ndarray, cy: float, cx: float
) -> tuple[float, float, float]:
    yy, xx = np.indices(signal.shape, dtype=np.float64)
    weights = np.where(mask, np.clip(signal, 0.0, None), 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Non-positive total flux inside mask")

    y = yy - cy
    x = xx - cx
    mu_xx = float((weights * x * x).sum() / total)
    mu_yy = float((weights * y * y).sum() / total)
    mu_xy = float((weights * x * y).sum() / total)
    cov = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(np.sort(eigvals)[::-1], 1e-8, None)
    major_sigma = float(np.sqrt(eigvals[0]))
    minor_sigma = float(np.sqrt(eigvals[1]))
    axis_ratio = float(np.clip(minor_sigma / major_sigma, 0.0, 1.0))
    ellipticity = float(1.0 - axis_ratio)
    return ellipticity, major_sigma, minor_sigma


def build_context(image_path: str, min_area: int = 32) -> ImageContext:
    rgb = load_image_rgb(image_path)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    signal = robust_background_subtract(gray)
    mask = largest_component_mask(signal, min_area=min_area)
    cy, cx = weighted_centroid(signal, mask)
    ellipticity, major_sigma, minor_sigma = second_moments(signal, mask, cy, cx)
    return ImageContext(
        path=os.path.abspath(image_path),
        rgb=rgb,
        gray=gray,
        signal=signal,
        mask=mask,
        centroid_y=cy,
        centroid_x=cx,
        area=int(mask.sum()),
        ellipticity=ellipticity,
        major_sigma=major_sigma,
        minor_sigma=minor_sigma,
    )


def radial_distances(shape: tuple[int, int], cy: float, cx: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    return np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)


def cumulative_radii(
    signal: np.ndarray, mask: np.ndarray, cy: float, cx: float
) -> tuple[float, float, float]:
    radii = radial_distances(signal.shape, cy, cx)[mask]
    flux = np.clip(signal[mask], 0.0, None).astype(np.float64)
    if flux.size == 0 or float(flux.sum()) <= 0.0:
        raise ValueError("No valid flux for cumulative radius calculation")
    order = np.argsort(radii)
    radii_sorted = radii[order]
    flux_sorted = flux[order]
    cum_flux = np.cumsum(flux_sorted)
    total = float(cum_flux[-1])
    r20 = float(np.interp(0.2 * total, cum_flux, radii_sorted))
    r50 = float(np.interp(0.5 * total, cum_flux, radii_sorted))
    r80 = float(np.interp(0.8 * total, cum_flux, radii_sorted))
    return r20, r50, r80


def concentration_index(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    r20, _, r80 = cumulative_radii(signal, mask, cy, cx)
    if r20 <= 0.0 or r80 <= 0.0:
        return None
    return float(5.0 * math.log10(r80 / r20))


def rotate_180_about_center(image: np.ndarray, cy: float, cx: float, order: int = 1) -> np.ndarray:
    h, w = image.shape[:2]
    target_cy = (h - 1) / 2.0
    target_cx = (w - 1) / 2.0
    shift_to_center = (target_cy - cy, target_cx - cx)

    if image.ndim == 2:
        shifted = ndimage.shift(
            image, shift=shift_to_center, order=order, mode="constant", cval=0.0
        )
        rotated = shifted[::-1, ::-1]
        restored = ndimage.shift(
            rotated,
            shift=(-shift_to_center[0], -shift_to_center[1]),
            order=order,
            mode="constant",
            cval=0.0,
        )
        return restored

    channels = []
    for idx in range(image.shape[2]):
        shifted = ndimage.shift(
            image[..., idx],
            shift=shift_to_center,
            order=order,
            mode="constant",
            cval=0.0,
        )
        rotated = shifted[::-1, ::-1]
        restored = ndimage.shift(
            rotated,
            shift=(-shift_to_center[0], -shift_to_center[1]),
            order=order,
            mode="constant",
            cval=0.0,
        )
        channels.append(restored)
    return np.stack(channels, axis=-1)


def asymmetry(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    masked_signal = np.where(mask, signal, 0.0)
    if float(np.abs(masked_signal[mask]).sum()) <= 0.0:
        return None

    best_value: float | None = None
    for dy in (-1.0, 0.0, 1.0):
        for dx in (-1.0, 0.0, 1.0):
            rotated_signal = rotate_180_about_center(masked_signal, cy + dy, cx + dx, order=1)
            rotated_mask = rotate_180_about_center(
                mask.astype(np.float32), cy + dy, cx + dx, order=0
            )
            union_mask = mask | (rotated_mask >= 0.5)
            denom = float(np.abs(masked_signal[union_mask]).sum())
            if denom <= 0.0:
                continue
            value = float(np.abs(masked_signal - rotated_signal)[union_mask].sum() / denom)
            if best_value is None or value < best_value:
                best_value = value
    return best_value


def smoothness(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    _, r50, _ = cumulative_radii(signal, mask, cy, cx)
    sigma = max(1.0, 0.25 * r50)
    blurred = ndimage.gaussian_filter(signal, sigma=sigma)
    residual = np.clip(signal - blurred, 0.0, None)
    radii = radial_distances(signal.shape, cy, cx)
    annulus = mask & (radii >= 0.25 * r50)
    denom = float(np.clip(signal[annulus], 0.0, None).sum())
    if denom <= 0.0:
        return None
    return float(residual[annulus].sum() / denom)


def gini_coefficient(signal: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.clip(signal[mask], 0.0, None).astype(np.float64)
    values = values[values > 0]
    if values.size < 2:
        return None
    values.sort()
    n = values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    mean_val = float(values.mean())
    if mean_val <= 0.0:
        return None
    gini = ((2.0 * index - n - 1.0) * values).sum() / (n * (n - 1) * mean_val)
    return float(gini)


def m20_statistic(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    yy, xx = np.indices(signal.shape, dtype=np.float64)
    flux = np.clip(signal, 0.0, None)
    flux = np.where(mask, flux, 0.0).astype(np.float64)
    valid = flux > 0
    if not bool(np.any(valid)):
        return None

    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    total_moment = float((flux[valid] * dist2[valid]).sum())
    if total_moment <= 0.0:
        return None

    flux_values = flux[valid]
    moment_values = (flux * dist2)[valid]
    order = np.argsort(flux_values)[::-1]
    flux_sorted = flux_values[order]
    moment_sorted = moment_values[order]
    cum_flux = np.cumsum(flux_sorted)
    threshold = 0.2 * float(flux_sorted.sum())
    idx = int(np.searchsorted(cum_flux, threshold, side="left")) + 1
    idx = min(idx, moment_sorted.size)
    top20_moment = float(moment_sorted[:idx].sum())
    return float(math.log10(max(top20_moment, 1e-12) / total_moment))


def radial_profile(
    signal: np.ndarray, mask: np.ndarray, cy: float, cx: float
) -> tuple[np.ndarray, np.ndarray]:
    radii = radial_distances(signal.shape, cy, cx)[mask]
    flux = np.clip(signal[mask], 0.0, None).astype(np.float64)
    valid = flux > 0
    radii = radii[valid]
    flux = flux[valid]
    if flux.size < 16:
        raise ValueError("Too few valid pixels for radial-profile fitting")

    rmax = float(np.percentile(radii, 99.5))
    bins = np.linspace(0.0, max(rmax, 2.0), num=min(40, max(16, int(np.sqrt(flux.size)))))
    bin_ids = np.digitize(radii, bins)

    r_centers: list[float] = []
    prof_vals: list[float] = []
    for bin_idx in range(1, len(bins)):
        in_bin = bin_ids == bin_idx
        if np.count_nonzero(in_bin) < 8:
            continue
        r_centers.append(float(np.median(radii[in_bin])))
        prof_vals.append(float(np.median(flux[in_bin])))

    r = np.asarray(r_centers, dtype=np.float64)
    prof = np.asarray(prof_vals, dtype=np.float64)
    positive = prof > 0
    r = r[positive]
    prof = prof[positive]
    if prof.size < 8:
        raise ValueError("Radial profile is too sparse for fitting")
    return r, prof


def sersic_profile(radius: np.ndarray, ie: float, re: float, n: float) -> np.ndarray:
    n = np.clip(n, 0.3, 8.0)
    re = np.clip(re, 1e-3, None)
    bn = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    return ie * np.exp(-bn * ((radius / re) ** (1.0 / n) - 1.0))


def fit_sersic_n(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    try:
        radius, profile = radial_profile(signal, mask, cy, cx)
    except ValueError:
        return None

    try:
        _, r50, _ = cumulative_radii(signal, mask, cy, cx)
        p0 = [float(np.max(profile)), max(float(r50), 1.0), 2.0]
        bounds = (
            [1e-6, 0.5, 0.3],
            [10.0 * float(np.max(profile)) + 1e-6, float(radius.max()) * 2.0 + 2.0, 8.0],
        )
        params, _ = curve_fit(sersic_profile, radius, profile, p0=p0, bounds=bounds, maxfev=20000)
        return float(params[2])
    except Exception:
        return None


def composite_profile(
    radius: np.ndarray, ib: float, re_b: float, idisk: float, h: float
) -> np.ndarray:
    bulge = sersic_profile(radius, ib, re_b, 4.0)
    disk = idisk * np.exp(-radius / np.clip(h, 1e-3, None))
    return bulge + disk


def sersic_total_flux(ie: float, re: float, n: float) -> float:
    bn = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    total = 2.0 * math.pi * ie * (re**2) * n * math.exp(bn) * gamma(2.0 * n) / (bn ** (2.0 * n))
    return float(total)


def fit_bulge_to_total(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    try:
        radius, profile = radial_profile(signal, mask, cy, cx)
    except ValueError:
        return None

    try:
        _, r50, _ = cumulative_radii(signal, mask, cy, cx)
        p0 = [
            float(profile.max()),
            max(0.5 * float(r50), 1.0),
            float(profile.max()) * 0.3,
            max(float(r50), 1.5),
        ]
        bounds = (
            [1e-6, 0.3, 1e-6, 0.3],
            [
                10.0 * float(profile.max()) + 1e-6,
                float(radius.max()) * 1.5 + 2.0,
                10.0 * float(profile.max()) + 1e-6,
                float(radius.max()) * 3.0 + 3.0,
            ],
        )
        params, _ = curve_fit(
            composite_profile, radius, profile, p0=p0, bounds=bounds, maxfev=20000
        )
        ib, re_b, idisk, h = [float(x) for x in params]
        bulge_flux = sersic_total_flux(ib, re_b, 4.0)
        disk_flux = float(2.0 * math.pi * idisk * h * h)
        total_flux = bulge_flux + disk_flux
        if total_flux <= 0.0:
            return None
        return float(np.clip(bulge_flux / total_flux, 0.0, 1.0))
    except Exception:
        return None


def shape_asymmetry(mask: np.ndarray, cy: float, cx: float) -> float | None:
    mask_f = mask.astype(np.float32)
    rotated = rotate_180_about_center(mask_f, cy, cx, order=0)
    rotated = rotated >= 0.5
    union = mask | rotated
    denom = float(union.sum())
    if denom <= 0.0:
        return None
    return float(np.logical_xor(mask, rotated).sum() / denom)


def multimode_statistic(signal: np.ndarray, mask: np.ndarray) -> float | None:
    smooth = ndimage.gaussian_filter(np.where(mask, signal, 0.0), sigma=1.0)
    values = smooth[mask]
    values = values[values > 0]
    if values.size < 16:
        return None

    total_area = float(mask.sum())
    best = 0.0
    for percentile in np.linspace(50, 95, num=10):
        threshold = float(np.percentile(values, percentile))
        candidate = (smooth >= threshold) & mask
        labeled = label(candidate)
        if labeled.max() < 2:
            continue
        _, counts = np.unique(labeled[labeled > 0], return_counts=True)
        counts = np.sort(counts)[::-1]
        if counts[0] <= 0:
            continue
        a1 = float(counts[0])
        a2 = float(counts[1])
        score = (a2 / a1) * (a2 / total_area)
        best = max(best, float(score))
    return float(best)


def intensity_statistic(signal: np.ndarray, mask: np.ndarray) -> float | None:
    smooth = ndimage.gaussian_filter(np.where(mask, signal, 0.0), sigma=1.0)
    maxima = smooth == ndimage.maximum_filter(smooth, size=5)
    maxima &= mask
    peaks = np.sort(smooth[maxima])[::-1]
    peaks = peaks[peaks > 0]
    if peaks.size == 0:
        return None
    if peaks.size == 1:
        return 0.0
    return float(np.clip(peaks[1] / max(peaks[0], 1e-12), 0.0, 1.0))


def deviation_statistic(signal: np.ndarray, mask: np.ndarray, cy: float, cx: float) -> float | None:
    smooth = ndimage.gaussian_filter(np.where(mask, signal, 0.0), sigma=1.0)
    values = smooth[mask]
    if values.size == 0:
        return None
    threshold = float(np.percentile(values[values > 0], 80)) if np.any(values > 0) else 0.0
    bright = (smooth >= threshold) & mask
    labeled = label(bright)
    if labeled.max() == 0:
        peak_y, peak_x = np.unravel_index(np.argmax(smooth), smooth.shape)
        by, bx = float(peak_y), float(peak_x)
    else:
        ids, counts = np.unique(labeled[labeled > 0], return_counts=True)
        best_id = int(ids[np.argmax(counts)])
        brightest_component = labeled == best_id
        by, bx = weighted_centroid(smooth, brightest_component)

    equiv_radius = math.sqrt(float(mask.sum()) / math.pi)
    if equiv_radius <= 0.0:
        return None
    dist = math.sqrt((by - cy) ** 2 + (bx - cx) ** 2)
    return float(dist / equiv_radius)


def color_gradient(
    rgb: np.ndarray, mask: np.ndarray, cy: float, cx: float, r50: float
) -> float | None:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    radii = radial_distances(mask.shape, cy, cx)
    inner = mask & (radii <= max(r50, 1.0))
    outer = mask & (radii > max(r50, 1.0)) & (radii <= max(2.0 * r50, r50 + 1.0))
    if int(inner.sum()) < 4 or int(outer.sum()) < 4:
        return None

    red = rgb[..., 0].astype(np.float64)
    blue = rgb[..., 2].astype(np.float64)
    color_map = (red - blue) / (red + blue + 1e-6)
    inner_mean = float(np.mean(color_map[inner]))
    outer_mean = float(np.mean(color_map[outer]))
    return float(inner_mean - outer_mean)


def image_diagonal(shape: tuple[int, int]) -> float:
    h, w = shape
    return float(math.sqrt(float(h * h + w * w)))


def reset_sanitize_statistics(preview_limit: int = DEFAULT_FAILURE_PREVIEW_LIMIT) -> None:
    global SANITIZE_EXAMPLE_LIMIT
    SANITIZE_COUNTS.clear()
    SANITIZE_EXAMPLES.clear()
    SANITIZE_EXAMPLE_LIMIT = max(0, int(preview_limit))


def _example_label(source_path: str) -> str:
    return os.path.basename(source_path) if source_path else "<unknown>"


def _record_sanitize_event(
    *,
    kind: str,
    name: str,
    raw_value: Any,
    new_value: float | int | None,
    source_path: str,
) -> None:
    key = f"{kind}:{name}"
    SANITIZE_COUNTS[key] += 1
    examples = SANITIZE_EXAMPLES[key]
    if len(examples) >= SANITIZE_EXAMPLE_LIMIT:
        return
    if new_value is None:
        examples.append(f"{_example_label(source_path)} raw={raw_value!r}")
    else:
        examples.append(f"{_example_label(source_path)} raw={raw_value!r} -> {new_value!r}")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (np.floating, float, np.integer, int)):
        value_f = float(value)
        if not np.isfinite(value_f):
            return None
        return value_f
    return None


def _is_invalid_sentinel(value: float) -> bool:
    return any(abs(value - sentinel) <= 1e-12 for sentinel in INVALID_SENTINELS)


def domain_with_upper(domain: ValueDomain, upper: float | None) -> ValueDomain:
    if upper is None or not np.isfinite(float(upper)):
        return domain
    upper_f = float(upper)
    if domain.upper is not None:
        upper_f = min(float(domain.upper), upper_f)
    return ValueDomain(lower=domain.lower, upper=upper_f)


def sanitize_number(
    name: str,
    value: Any,
    *,
    domain: ValueDomain,
    source_path: str = "",
    integer: bool = False,
) -> float | int | None:
    if value is None:
        return None

    value_f = _finite_float(value)
    if value_f is None or _is_invalid_sentinel(value_f):
        _record_sanitize_event(
            kind="invalid",
            name=name,
            raw_value=value,
            new_value=None,
            source_path=source_path,
        )
        return None

    clipped = value_f
    if domain.lower is not None and clipped < float(domain.lower):
        clipped = float(domain.lower)
    if domain.upper is not None and clipped > float(domain.upper):
        clipped = float(domain.upper)

    if clipped != value_f:
        new_value: float | int = int(round(clipped)) if integer else float(clipped)
        _record_sanitize_event(
            kind="clipped",
            name=name,
            raw_value=value,
            new_value=new_value,
            source_path=source_path,
        )
        return new_value

    if integer:
        return int(round(clipped))
    return float(clipped)


def sanitize_radii(
    r20: Any,
    r50: Any,
    r80: Any,
    *,
    diagonal: float,
    source_path: str,
) -> tuple[float | None, float | None, float | None]:
    radius_domain = domain_with_upper(AUXILIARY_BASE_DOMAINS["r20"], diagonal)
    r20_s = sanitize_number("r20", r20, domain=radius_domain, source_path=source_path)
    r50_s = sanitize_number("r50", r50, domain=radius_domain, source_path=source_path)
    r80_s = sanitize_number("r80", r80, domain=radius_domain, source_path=source_path)
    if r20_s is None or r50_s is None or r80_s is None:
        return None, None, None
    if not (float(r20_s) <= float(r50_s) <= float(r80_s)):
        for name, value in (("r20", r20_s), ("r50", r50_s), ("r80", r80_s)):
            _record_sanitize_event(
                kind="invalid",
                name=name,
                raw_value=value,
                new_value=None,
                source_path=source_path,
            )
        return None, None, None
    return float(r20_s), float(r50_s), float(r80_s)


def optional_feature_value(name: str, source_path: str, compute_fn) -> Any:
    try:
        return compute_fn()
    except Exception as exc:
        _record_sanitize_event(
            kind="invalid",
            name=name,
            raw_value=f"{type(exc).__name__}: {exc}",
            new_value=None,
            source_path=source_path,
        )
        return None


def compute_feature_record(ctx: ImageContext) -> dict[str, Any]:
    h, w = ctx.signal.shape
    diagonal = image_diagonal(ctx.signal.shape)
    r20_raw, r50_raw, r80_raw = optional_feature_value(
        "cumulative_radii",
        ctx.path,
        lambda: cumulative_radii(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x),
    ) or (None, None, None)
    r20, r50, r80 = sanitize_radii(
        r20_raw,
        r50_raw,
        r80_raw,
        diagonal=diagonal,
        source_path=ctx.path,
    )

    if r20 is not None and r80 is not None and r20 > 0.0 and r80 > 0.0:
        concentration_value = 5.0 * math.log10(r80 / r20)
    else:
        concentration_value = None

    raw_features = {
        "C": concentration_value,
        "A": optional_feature_value(
            "A", ctx.path, lambda: asymmetry(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x)
        ),
        "S": optional_feature_value(
            "S", ctx.path, lambda: smoothness(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x)
        ),
        "G": optional_feature_value("G", ctx.path, lambda: gini_coefficient(ctx.signal, ctx.mask)),
        "M20": optional_feature_value(
            "M20",
            ctx.path,
            lambda: m20_statistic(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x),
        ),
        "ellipticity": ctx.ellipticity,
        "sersic_n": optional_feature_value(
            "sersic_n",
            ctx.path,
            lambda: fit_sersic_n(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x),
        ),
        "half_light_radius": r50,
        "B/T": optional_feature_value(
            "B/T",
            ctx.path,
            lambda: fit_bulge_to_total(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x),
        ),
        "shape_asymmetry": optional_feature_value(
            "shape_asymmetry",
            ctx.path,
            lambda: shape_asymmetry(ctx.mask, ctx.centroid_y, ctx.centroid_x),
        ),
        "MID_M": optional_feature_value(
            "MID_M", ctx.path, lambda: multimode_statistic(ctx.signal, ctx.mask)
        ),
        "MID_I": optional_feature_value(
            "MID_I", ctx.path, lambda: intensity_statistic(ctx.signal, ctx.mask)
        ),
        "MID_D": optional_feature_value(
            "MID_D",
            ctx.path,
            lambda: deviation_statistic(ctx.signal, ctx.mask, ctx.centroid_y, ctx.centroid_x),
        ),
        "color_gradient": (
            optional_feature_value(
                "color_gradient",
                ctx.path,
                lambda: color_gradient(ctx.rgb, ctx.mask, ctx.centroid_y, ctx.centroid_x, r50),
            )
            if r50 is not None
            else None
        ),
    }
    feature_domains = {
        **FEATURE_DOMAINS,
        "half_light_radius": domain_with_upper(FEATURE_DOMAINS["half_light_radius"], diagonal),
    }
    features = {
        name: sanitize_number(
            name,
            raw_features.get(name),
            domain=feature_domains[name],
            source_path=ctx.path,
        )
        for name in FEATURE_NAMES
    }

    major_sigma = sanitize_number(
        "major_sigma",
        ctx.major_sigma,
        domain=domain_with_upper(AUXILIARY_BASE_DOMAINS["major_sigma"], diagonal),
        source_path=ctx.path,
    )
    minor_sigma = sanitize_number(
        "minor_sigma",
        ctx.minor_sigma,
        domain=domain_with_upper(AUXILIARY_BASE_DOMAINS["minor_sigma"], diagonal),
        source_path=ctx.path,
    )
    if major_sigma is not None and minor_sigma is not None and minor_sigma > major_sigma:
        _record_sanitize_event(
            kind="clipped",
            name="major_sigma",
            raw_value=major_sigma,
            new_value=minor_sigma,
            source_path=ctx.path,
        )
        _record_sanitize_event(
            kind="clipped",
            name="minor_sigma",
            raw_value=minor_sigma,
            new_value=major_sigma,
            source_path=ctx.path,
        )
        major_sigma, minor_sigma = minor_sigma, major_sigma

    auxiliary = {
        "mask_area": sanitize_number(
            "mask_area",
            ctx.area,
            domain=domain_with_upper(AUXILIARY_BASE_DOMAINS["mask_area"], float(h * w)),
            source_path=ctx.path,
            integer=True,
        ),
        "centroid_y": sanitize_number(
            "centroid_y",
            ctx.centroid_y,
            domain=domain_with_upper(AUXILIARY_BASE_DOMAINS["centroid_y"], float(max(h - 1, 0))),
            source_path=ctx.path,
        ),
        "centroid_x": sanitize_number(
            "centroid_x",
            ctx.centroid_x,
            domain=domain_with_upper(AUXILIARY_BASE_DOMAINS["centroid_x"], float(max(w - 1, 0))),
            source_path=ctx.path,
        ),
        "r20": r20,
        "r50": r50,
        "r80": r80,
        "major_sigma": major_sigma,
        "minor_sigma": minor_sigma,
    }

    record = {
        "image_name": os.path.basename(ctx.path),
        "path": ctx.path,
        "status": "ok",
        "error_message": None,
        "features": features,
        "auxiliary": auxiliary,
    }
    return record


def compute_record(image_path: str, min_area: int = 32) -> dict[str, Any]:
    try:
        ctx = build_context(image_path, min_area=min_area)
        return compute_feature_record(ctx)
    except Exception as exc:
        return {
            "image_name": os.path.basename(image_path),
            "path": os.path.abspath(image_path),
            "status": "failed",
            "error_message": str(exc),
            "features": {
                "C": None,
                "A": None,
                "S": None,
                "G": None,
                "M20": None,
                "ellipticity": None,
                "sersic_n": None,
                "half_light_radius": None,
                "B/T": None,
                "shape_asymmetry": None,
                "MID_M": None,
                "MID_I": None,
                "MID_D": None,
                "color_gradient": None,
            },
            "auxiliary": None,
        }


def collect_failure_statistics(
    records: list[dict[str, Any]], *, preview_limit: int
) -> dict[str, Any]:
    failed_images: list[dict[str, str]] = []
    failed_image_count = 0
    feature_failure_counts: Counter[str] = Counter()
    feature_failure_examples: dict[str, list[str]] = {name: [] for name in FEATURE_NAMES}
    preview_limit = max(0, int(preview_limit))

    for record in records:
        image_name = str(record.get("image_name", ""))
        image_path = str(record.get("path", ""))
        display_name = image_name or image_path

        if record.get("status") != "ok":
            failed_image_count += 1
            if len(failed_images) < preview_limit:
                failed_images.append(
                    {
                        "image_name": image_name,
                        "path": image_path,
                        "error_message": str(record.get("error_message", "")),
                    }
                )

        features = record.get("features") or {}
        for feature_name in FEATURE_NAMES:
            if features.get(feature_name) is None:
                feature_failure_counts[feature_name] += 1
                examples = feature_failure_examples[feature_name]
                if len(examples) < preview_limit:
                    examples.append(display_name)

    return {
        "failed_image_count": failed_image_count,
        "failed_images": failed_images,
        "feature_failure_counts": feature_failure_counts,
        "feature_failure_examples": feature_failure_examples,
    }


def print_summary(
    *,
    logger: logging.Logger,
    total_found: int,
    skipped_existing: int,
    newly_processed: int,
    succeeded: int,
    failed: int,
    output_json: str,
    log_path: str,
    processed_records: list[dict[str, Any]],
    failure_preview_limit: int,
) -> None:
    stats = collect_failure_statistics(
        processed_records,
        preview_limit=failure_preview_limit,
    )

    logger.info("========== Morphology extraction summary ==========")
    logger.info("Total discovered images : %d", total_found)
    logger.info("Skipped existing        : %d", skipped_existing)
    logger.info("Newly processed         : %d", newly_processed)
    logger.info("Succeeded               : %d", succeeded)
    logger.info("Failed                  : %d", failed)
    logger.info("Saved JSON              : %s", os.path.abspath(output_json))
    logger.info("Saved log               : %s", os.path.abspath(log_path))

    failed_images = stats["failed_images"]
    failed_image_count = int(stats["failed_image_count"])
    feature_failure_counts = stats["feature_failure_counts"]
    feature_failure_examples = stats["feature_failure_examples"]

    logger.info("---------- Failed image summary ----------")
    logger.info("Failed image count      : %d", failed_image_count)
    if failed_images:
        logger.info("Failed image preview    : first %d", len(failed_images))
        for idx, item in enumerate(failed_images, start=1):
            logger.info(
                "[%d] image=%s | path=%s | error=%s",
                idx,
                item["image_name"],
                item["path"],
                item["error_message"],
            )

    logger.info("---------- Feature failure summary ----------")
    for feature_name in FEATURE_NAMES:
        count = int(feature_failure_counts[feature_name])
        names = feature_failure_examples[feature_name]
        logger.info("%s is None on %d image(s)", feature_name, count)
        if names:
            logger.info("%s None preview: %s", feature_name, ", ".join(names))

    logger.info("---------- Domain sanitization summary ----------")
    if not SANITIZE_COUNTS:
        logger.info("No invalid finite values or out-of-domain values were sanitized.")
    for key in sorted(SANITIZE_COUNTS):
        logger.info("%s count=%d", key, int(SANITIZE_COUNTS[key]))
        examples = SANITIZE_EXAMPLES.get(key, [])
        if examples:
            logger.info("%s preview: %s", key, "; ".join(examples))

    logger.info("==================================================")


def main() -> None:
    args = build_argparser().parse_args()
    failure_preview_limit = max(0, int(args.failure_preview_limit))
    reset_sanitize_statistics(preview_limit=failure_preview_limit)

    input_path_obj = Path(args.input_path)
    default_output_dir = input_path_obj if input_path_obj.is_dir() else input_path_obj.parent

    output_json = args.output_json
    if not output_json:
        output_json = str(default_output_dir / "structural.json")

    log_path = args.log_path
    if not log_path:
        log_path = str(default_output_dir / "morphology_features.log")

    logger = build_logger(log_path)

    image_paths = iter_image_files(args.input_path, recursive=args.recursive)
    existing_records, existing_map = load_existing_records(output_json)
    records_map = dict(existing_map)

    skipped_existing = 0
    to_process: list[str] = []
    for image_path in image_paths:
        image_path = os.path.abspath(image_path)
        if (not args.overwrite) and image_path in records_map:
            skipped_existing += 1
            continue
        to_process.append(image_path)

    succeeded = 0
    failed = 0
    new_since_save = 0
    processed_records: list[dict[str, Any]] = []

    iterator = tqdm(to_process, desc="Computing morphology", unit="image")
    for image_path in iterator:
        record = compute_record(image_path, min_area=args.min_area)
        records_map[image_path] = record
        processed_records.append(record)
        if record["status"] == "ok":
            succeeded += 1
        else:
            failed += 1
        new_since_save += 1

        if args.save_every > 0 and new_since_save >= args.save_every:
            save_records(output_json, list(records_map.values()), indent=args.indent)
            new_since_save = 0

    save_records(output_json, list(records_map.values()), indent=args.indent)

    print_summary(
        logger=logger,
        total_found=len(image_paths),
        skipped_existing=skipped_existing,
        newly_processed=len(to_process),
        succeeded=succeeded,
        failed=failed,
        output_json=output_json,
        log_path=log_path,
        processed_records=processed_records,
        failure_preview_limit=failure_preview_limit,
    )


if __name__ == "__main__":
    main()
