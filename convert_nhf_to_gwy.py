#!/usr/bin/env python3
"""
Convert NanoSurf .nhf HDF5 files to Gwyddion .gwy files and create overview PDFs.

First-time setup from the folder where you want to keep the script:

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install h5py numpy matplotlib gwyfile
    python3 convert_nhf_to_gwy.py

Next times:

    source .venv/bin/activate
    python3 convert_nhf_to_gwy.py

Optional direct path usage:

    python3 convert_nhf_to_gwy.py /path/to/file_or_folder
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

try:
    from gwyfile.objects import GwyContainer, GwyDataField, GwySIUnit
except Exception as exc:  
    GwyContainer = None
    GwyDataField = None
    GwySIUnit = None
    GWY_IMPORT_ERROR = exc
else:
    GWY_IMPORT_ERROR = None


# Edit this default path once if desired. The script still asks for confirmation.
DEFAULT_INPUT_PATH = Path("/path/to/your/folder")
# If True, existing .gwy and overview PDF files are overwritten.
OVERWRITE_EXISTING = True

# Maximum number of channel groups shown on one PDF page.
# A channel group can contain one image or a forward/backward pair.
# Four groups per page gives a two-page overview for the usual NanoSurf files
# with seven logical channels.
CHANNEL_GROUPS_PER_PDF_PAGE = 4

# Fixed PDF layout. These values are in inches for the page and in normalized
# figure coordinates for positions. Keeping them fixed is what guarantees that
# all image panels are square and equally sized across the whole PDF.
PDF_FIG_WIDTH = 16.0
PDF_FIG_HEIGHT = 10.4
PDF_GRID_NROWS = 2
PDF_GRID_NCOLS = 2

# University of Basel inspired colour gradient for the overview PDF previews.
UNIBAS_COLORS = [
    (0.00, "#2D373C"),  # Unibas Anthrazit
    (0.20, "#46505A"),  # Unibas Anthrazit hell
    (0.45, "#A5D7D2"),  # Unibas Mint
    (0.65, "#D2EBE9"),  # Unibas Mint hell
    (0.82, "#FFFFFF"),  # Weiss
    (1.00, "#D20537"),  # Unibas Rot
]
UNIBAS_CMAP = LinearSegmentedColormap.from_list("Unibas", UNIBAS_COLORS, N=256)


@dataclass
class Channel:
    """One image channel extracted from the NHF/HDF5 file."""

    h5_path: str
    title: str
    base_title: str
    direction: str
    group_key: str
    data: np.ndarray
    xreal: float
    yreal: float
    xy_unit: str
    z_unit: str


@dataclass
class ChannelGroup:
    """One logical AFM channel, optionally containing forward and backward images."""

    key: str
    title: str
    channels: List[Channel]


@dataclass
class ConversionResult:
    """Conversion result for one .nhf file."""

    nhf_path: Path
    gwy_path: Optional[Path]
    channels: List[Channel]
    metadata: Dict[str, Any]
    error: Optional[str] = None


def decode_value(value: Any) -> Any:
    """Convert HDF5 attribute values into readable Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O"}:
            return [decode_value(v) for v in value.tolist()]
        if value.size == 1:
            return decode_value(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attrs_to_dict(h5obj: Any) -> Dict[str, Any]:
    """Read all attributes of an HDF5 object as a normal dictionary."""
    out: Dict[str, Any] = {}
    for key, value in h5obj.attrs.items():
        out[str(key)] = decode_value(value)
    return out


def clean_text(value: Any) -> str:
    """Return a compact text representation for metadata values."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(clean_text(v) for v in value)
    text = str(value).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def is_uuid_like(text: str) -> bool:
    """Avoid using UUID strings as channel names when better labels exist."""
    text = text.strip().lower()
    return len(text) == 36 and text.count("-") == 4


def compact_unit(unit_text: str) -> str:
    """Convert common verbose unit strings into short labels."""
    text = unit_text.strip().lower()
    if "meter" in text or text == "m":
        return "m"
    if "volt" in text or text == "v":
        return "V"
    if "hertz" in text or text == "hz":
        return "Hz"
    if "ampere" in text or text == "a":
        return "A"
    if "degree" in text:
        return "deg"
    if "radian" in text:
        return "rad"
    return unit_text.strip()


def find_first_attr_containing(attrs: Dict[str, Any], keywords: Sequence[str]) -> Optional[Any]:
    """Find the first attribute whose key contains one of the requested keywords."""
    for key, value in attrs.items():
        low_key = key.lower()
        if any(word in low_key for word in keywords):
            return value
    return None


def infer_scan_size(file_obj: h5py.File) -> Tuple[float, float, str]:
    """Extract lateral scan size from metadata when possible."""
    candidates: List[Dict[str, Any]] = []

    def collect_attrs(_name: str, obj: Any) -> None:
        attrs = attrs_to_dict(obj)
        if attrs:
            candidates.append(attrs)

    file_obj.visititems(collect_attrs)
    candidates.insert(0, attrs_to_dict(file_obj))

    for attrs in candidates:
        for key, value in attrs.items():
            low_key = key.lower()
            if "rect_axis_range" in low_key or "scan_range" in low_key or "axis_range" in low_key:
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    try:
                        return float(value[0]), float(value[1]), "m"
                    except Exception:
                        pass

    return 1.0, 1.0, "m"


def infer_image_shape(data: np.ndarray, dataset_attrs: Dict[str, Any], parent_attrs: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Infer the 2D image shape from the dataset size and metadata."""
    arr = np.asarray(data)

    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])

    if arr.ndim != 1 or arr.size == 0:
        return None

    attrs = {**parent_attrs, **dataset_attrs}
    lower_attrs = {key.lower(): value for key, value in attrs.items()}

    possible_pairs = [
        ("xres", "yres"),
        ("x_res", "y_res"),
        ("x-pixels", "y-pixels"),
        ("x_pixels", "y_pixels"),
        ("xpixels", "ypixels"),
        ("width_pixels", "height_pixels"),
        ("points_x", "points_y"),
        ("columns", "rows"),
    ]

    for x_key, y_key in possible_pairs:
        if x_key in lower_attrs and y_key in lower_attrs:
            try:
                xres = int(lower_attrs[x_key])
                yres = int(lower_attrs[y_key])
                if xres > 1 and yres > 1 and xres * yres == arr.size:
                    return yres, xres
            except Exception:
                pass

    for value in attrs.values():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                a, b = int(value[0]), int(value[1])
                if a > 1 and b > 1 and a * b == arr.size:
                    return b, a
            except Exception:
                pass

    side = int(round(math.sqrt(arr.size)))
    if side * side == arr.size:
        return side, side

    return None


def normalize_channel_name(text: str) -> str:
    """Clean channel names used in titles and grouping."""
    text = clean_text(text)
    text = re.sub(r"^\d+\s*-\s*", "", text)
    text = re.sub(r"\s*\((forward|backward|trace|retrace|subgroup_\d+)\)\s*$", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text or "Channel"


def infer_channel_base_title(h5_path: str, dataset_attrs: Dict[str, Any], parent_attrs: Dict[str, Any]) -> str:
    """Build a readable logical channel title without forward/backward information."""
    attrs = {**parent_attrs, **dataset_attrs}
    lower_attrs = {key.lower(): value for key, value in attrs.items()}

    preferred_keys = [
        "name", "item_name", "signal_name", "channel_name", "channel", "quantity",
        "quantity_name", "data_name", "label", "long_name", "description",
    ]

    parts: List[str] = []
    for key in preferred_keys:
        if key in lower_attrs:
            text = normalize_channel_name(lower_attrs[key])
            if text and not is_uuid_like(text) and text not in parts:
                parts.append(text)

    for key, value in attrs.items():
        low_key = key.lower()
        if any(word in low_key for word in ["position", "deflection", "amplitude", "phase", "frequency", "bias", "height", "topo", "z_"]):
            text = normalize_channel_name(value)
            if text and not is_uuid_like(text) and text not in parts:
                parts.append(text)

    if not parts:
        path_title = h5_path.replace("/", "_")
        parts.append(normalize_channel_name(path_title))

    return " | ".join(parts[:3])


def infer_scan_direction(h5_path: str, parent_attrs: Dict[str, Any]) -> str:
    """Infer trace/retrace direction from NHF subgroup paths or metadata."""
    attrs_text = " ".join(clean_text(v).lower() for v in parent_attrs.values())
    if any(word in attrs_text for word in ["forward", "trace", "up"]):
        return "Forward"
    if any(word in attrs_text for word in ["backward", "retrace", "down"]):
        return "Backward"

    match = re.search(r"subgroup_(\d+)", h5_path)
    if match:
        subgroup_index = int(match.group(1))
        if subgroup_index == 0:
            return "Forward"
        if subgroup_index == 1:
            return "Backward"
        return f"Direction {subgroup_index}"

    return "Single"


def make_channel_group_key(h5_path: str, base_title: str) -> str:
    """Create a stable key that groups matching forward/backward channels."""
    path_key = re.sub(r"subgroup_\d+", "subgroup_*", h5_path)
    return f"{base_title}::{path_key}"


def infer_z_unit(dataset_attrs: Dict[str, Any], parent_attrs: Dict[str, Any]) -> str:
    """Infer vertical/signal unit from metadata when possible."""
    attrs = {**parent_attrs, **dataset_attrs}
    unit = find_first_attr_containing(attrs, ["unit"])
    if unit is None:
        return "arb."
    return compact_unit(clean_text(unit)) or "arb."


def apply_simple_linear_calibration(data: np.ndarray, attrs: Dict[str, Any]) -> np.ndarray:
    """Apply obvious scale/offset metadata when present; otherwise return raw data."""
    arr = np.asarray(data, dtype=np.float64)
    lower_attrs = {key.lower(): value for key, value in attrs.items()}

    scale = None
    offset = None
    for key in ["scale", "slope", "factor", "multiplier", "calibration_factor", "calibration_scale"]:
        if key in lower_attrs:
            try:
                scale = float(lower_attrs[key])
                break
            except Exception:
                pass

    for key in ["offset", "intercept", "calibration_offset"]:
        if key in lower_attrs:
            try:
                offset = float(lower_attrs[key])
                break
            except Exception:
                pass

    if scale is not None:
        arr = arr * scale
    if offset is not None:
        arr = arr + offset

    return arr


def read_nhf_channels(nhf_path: Path) -> Tuple[List[Channel], Dict[str, Any]]:
    """Read all non-empty numeric 2D-compatible datasets from one NHF file."""
    channels: List[Channel] = []
    metadata: Dict[str, Any] = {}

    with h5py.File(nhf_path, "r") as file_obj:
        metadata.update(attrs_to_dict(file_obj))
        xreal, yreal, xy_unit = infer_scan_size(file_obj)

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.size == 0:
                return
            if obj.dtype.kind not in {"i", "u", "f"}:
                return

            dataset_attrs = attrs_to_dict(obj)
            parent_attrs: Dict[str, Any] = {}
            parent_name = str(Path(name).parent).replace(".", "")
            if parent_name and parent_name in file_obj:
                parent_attrs = attrs_to_dict(file_obj[parent_name])

            raw = obj[()]
            shape = infer_image_shape(raw, dataset_attrs, parent_attrs)
            if shape is None:
                return

            yres, xres = shape
            if yres * xres != np.asarray(raw).size:
                return

            combined_attrs = {**parent_attrs, **dataset_attrs}
            image = apply_simple_linear_calibration(np.asarray(raw).reshape(yres, xres), combined_attrs)

            index = len(channels)
            base_title = infer_channel_base_title(name, dataset_attrs, parent_attrs)
            direction = infer_scan_direction(name, parent_attrs)
            group_key = make_channel_group_key(name, base_title)
            title = f"{index:02d} - {base_title}"
            if direction != "Single":
                title += f" ({direction})"

            channels.append(
                Channel(
                    h5_path=name,
                    title=title,
                    base_title=base_title,
                    direction=direction,
                    group_key=group_key,
                    data=image,
                    xreal=float(xreal),
                    yreal=float(yreal),
                    xy_unit=xy_unit,
                    z_unit=infer_z_unit(dataset_attrs, parent_attrs),
                )
            )

        file_obj.visititems(visit)

    return channels, metadata


def make_gwy_si_unit(unit: str) -> Optional[Any]:
    """Create a Gwyddion SI unit object when supported by gwyfile."""
    if GwySIUnit is None:
        return None
    try:
        return GwySIUnit(unit)
    except Exception:
        return None


def set_gwy_member(obj: Any, key: str, value: Any) -> None:
    """Set a member on a gwyfile object using both mapping and attribute styles when possible."""
    try:
        obj[key] = value
    except Exception:
        pass
    try:
        setattr(obj, key, value)
    except Exception:
        pass


def make_gwy_datafield(channel: Channel) -> Any:
    """Create a GwyDataField and preserve lateral scale and units."""
    if GwyDataField is None:
        raise RuntimeError(f"Could not import gwyfile: {GWY_IMPORT_ERROR}")

    arr = np.asarray(channel.data, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    yres, xres = arr.shape

    try:
        field = GwyDataField(arr)
    except TypeError:
        try:
            field = GwyDataField(data=arr)
        except TypeError:
            try:
                field = GwyDataField(xres, yres, float(channel.xreal), float(channel.yreal), arr)
            except TypeError:
                field = GwyDataField()

    # These dictionary members are what Gwyddion uses for the real scale.
    set_gwy_member(field, "xres", int(xres))
    set_gwy_member(field, "yres", int(yres))
    set_gwy_member(field, "xreal", float(channel.xreal))
    set_gwy_member(field, "yreal", float(channel.yreal))
    set_gwy_member(field, "data", arr)

    xy_unit = make_gwy_si_unit(channel.xy_unit)
    z_unit = make_gwy_si_unit(channel.z_unit)
    if xy_unit is not None:
        set_gwy_member(field, "si_unit_xy", xy_unit)
    if z_unit is not None:
        set_gwy_member(field, "si_unit_z", z_unit)

    return field


def write_gwy_file(channels: Sequence[Channel], output_path: Path) -> None:
    """Write all extracted channels into one Gwyddion .gwy file."""
    if GwyContainer is None:
        raise RuntimeError("Install the Python package 'gwyfile' with: python3 -m pip install gwyfile")

    container = GwyContainer()

    for idx, channel in enumerate(channels):
        field = make_gwy_datafield(channel)
        container[f"/{idx}/data"] = field
        container[f"/{idx}/data/title"] = channel.title
        container[f"/{idx}/meta/HDF5 path"] = channel.h5_path
        container[f"/{idx}/meta/Channel"] = channel.base_title
        container[f"/{idx}/meta/Direction"] = channel.direction
        container[f"/{idx}/meta/XY unit"] = channel.xy_unit
        container[f"/{idx}/meta/Z unit"] = channel.z_unit
        container[f"/{idx}/meta/X real"] = str(channel.xreal)
        container[f"/{idx}/meta/Y real"] = str(channel.yreal)

    container.tofile(str(output_path))


def robust_limits(data: np.ndarray) -> Tuple[float, float]:
    """Compute robust display limits for overview thumbnails."""
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if min_value == max_value:
        return min_value - 0.5, max_value + 0.5
    return float(np.nanpercentile(arr, 2)), float(np.nanpercentile(arr, 98))


def choose_length_display_unit(xreal: float, yreal: float, unit: str) -> Tuple[float, str]:
    """Choose a readable lateral unit for the PDF axes."""
    if unit != "m":
        return 1.0, unit

    size = max(abs(float(xreal)), abs(float(yreal)))
    if size < 1e-8:
        return 1e9, "nm"
    if size < 1e-4:
        return 1e6, "um"
    if size < 1e-1:
        return 1e3, "mm"
    return 1.0, "m"


def choose_signal_display_factor(data: np.ndarray, unit: str) -> Tuple[float, str]:
    """Choose a readable signal scaling factor and label for PDF colorbars."""
    arr = np.asarray(data, dtype=np.float64)
    finite = np.abs(arr[np.isfinite(arr)])
    ref = float(np.nanpercentile(finite, 95)) if finite.size else 0.0

    if unit == "m":
        if ref < 1e-10:
            return 1e12, "pm"
        if ref < 1e-7:
            return 1e9, "nm"
        if ref < 1e-4:
            return 1e6, "um"
        if ref < 1e-1:
            return 1e3, "mm"
        return 1.0, "m"

    if unit == "V":
        if ref < 1.0:
            return 1e3, "mV"
        return 1.0, "V"

    return 1.0, unit


def nice_number(value: float) -> float:
    """Round a length to a readable 1/2/5 × 10^n value."""
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.5:
        nice_fraction = 2.0
    elif fraction < 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def add_scale_bar(ax: plt.Axes, x_display: float, y_display: float, unit: str) -> None:
    """Add a simple physical scale bar to one PDF image panel."""
    if x_display <= 0 or y_display <= 0:
        return

    bar_length = nice_number(0.25 * x_display)
    x0 = 0.92 * x_display - bar_length
    x1 = x0 + bar_length
    y0 = 0.08 * y_display

    ax.plot([x0, x1], [y0, y0], color="white", linewidth=4, solid_capstyle="butt")
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5, solid_capstyle="butt")
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.03 * y_display,
        f"{bar_length:g} {unit}",
        ha="center",
        va="bottom",
        fontsize=7,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )


def direction_sort_key(channel: Channel) -> Tuple[int, str]:
    """Sort channels in the natural forward/backward order."""
    order = {"Forward": 0, "Backward": 1, "Single": 2}
    return order.get(channel.direction, 10), channel.h5_path


def build_channel_groups(channels: Sequence[Channel]) -> List[ChannelGroup]:
    """Group matching forward/backward channels for the overview PDF."""
    grouped: Dict[str, List[Channel]] = {}
    titles: Dict[str, str] = {}

    for channel in channels:
        grouped.setdefault(channel.group_key, []).append(channel)
        titles.setdefault(channel.group_key, channel.base_title)

    groups: List[ChannelGroup] = []
    for key, group_channels in grouped.items():
        group_channels = sorted(group_channels, key=direction_sort_key)
        groups.append(ChannelGroup(key=key, title=titles[key], channels=group_channels))

    def group_sort_key(group: ChannelGroup) -> Tuple[str, str]:
        first_path = group.channels[0].h5_path if group.channels else ""
        return group.title.lower(), first_path

    return sorted(groups, key=group_sort_key)


def draw_channel_group_at(fig: plt.Figure, cell: Tuple[float, float, float, float], group: ChannelGroup) -> None:
    """Draw one logical channel using fixed square image axes.

    The layout is computed in physical inches and only converted to Matplotlib
    normalized coordinates at the very end. This is important: a width and a
    height that look equal in normalized figure coordinates are not square when
    the page itself is rectangular. Computing the image size in inches
    guarantees true square panels in the exported PDF.
    """
    channels = sorted(group.channels, key=direction_sort_key)
    if not channels:
        return

    fig_width_in = float(fig.get_figwidth())
    fig_height_in = float(fig.get_figheight())

    cell_left, cell_bottom, cell_width, cell_height = cell
    cell_left_in = cell_left * fig_width_in
    cell_bottom_in = cell_bottom * fig_height_in
    cell_width_in = cell_width * fig_width_in
    cell_height_in = cell_height * fig_height_in


    pad_left_in = 0.12
    pad_right_in = 0.16
    pad_bottom_in = 0.10
    title_height_in = 0.26
    title_gap_in = 0.04
    image_gap_in = 0.14
    cbar_gap_in = 0.10
    cbar_width_in = 0.14

    usable_width_in = cell_width_in - pad_left_in - pad_right_in
    usable_height_in = cell_height_in - pad_bottom_in - title_height_in - title_gap_in

    max_square_from_width_in = (usable_width_in - image_gap_in - cbar_gap_in - cbar_width_in) / 2.0
    max_square_from_height_in = usable_height_in
    square_size_in = max(0.1, min(max_square_from_width_in, max_square_from_height_in))

    block_width_in = 2.0 * square_size_in + image_gap_in + cbar_gap_in + cbar_width_in
    block_left_in = cell_left_in + pad_left_in + 0.5 * (usable_width_in - block_width_in)
    image_bottom_in = cell_bottom_in + pad_bottom_in + 0.5 * (usable_height_in - square_size_in)

    def rect_in_to_norm(left_in: float, bottom_in: float, width_in: float, height_in: float) -> List[float]:
        return [left_in / fig_width_in, bottom_in / fig_height_in, width_in / fig_width_in, height_in / fig_height_in]

    title_ax = fig.add_axes(
        rect_in_to_norm(
            cell_left_in,
            cell_bottom_in + cell_height_in - title_height_in,
            cell_width_in,
            title_height_in,
        )
    )
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.45,
        group.title,
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
    )

    common_unit = channels[0].z_unit
    all_values = np.concatenate([np.asarray(ch.data, dtype=np.float64).ravel() for ch in channels])
    signal_scale, z_label = choose_signal_display_factor(all_values, common_unit)
    display_arrays = [np.asarray(ch.data, dtype=np.float64) * signal_scale for ch in channels]
    vmin, vmax = robust_limits(np.concatenate([arr.ravel() for arr in display_arrays]))

    first = channels[0]
    xy_scale, xy_label = choose_length_display_unit(first.xreal, first.yreal, first.xy_unit)
    x_display = first.xreal * xy_scale
    y_display = first.yreal * xy_scale

    image = None
    for idx in range(2):
        image_left_in = block_left_in + idx * (square_size_in + image_gap_in)
        ax = fig.add_axes(rect_in_to_norm(image_left_in, image_bottom_in, square_size_in, square_size_in))

        if idx >= len(channels):
            ax.axis("off")
            continue

        channel = channels[idx]
        data_display = display_arrays[idx]
        image = ax.imshow(
            data_display,
            origin="lower",
            extent=[0.0, x_display, 0.0, y_display],
            vmin=vmin,
            vmax=vmax,
            cmap=UNIBAS_CMAP,
            aspect="auto",
        )

        subtitle = channel.direction if channel.direction != "Single" else "Image"
        ax.set_title(subtitle, fontsize=7.8, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        add_scale_bar(ax, x_display, y_display, xy_label)

    cax_left_in = block_left_in + 2.0 * square_size_in + image_gap_in + cbar_gap_in
    cax = fig.add_axes(rect_in_to_norm(cax_left_in, image_bottom_in, cbar_width_in, square_size_in))
    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.set_title(z_label, fontsize=6.8, pad=4.0)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(labelsize=6.0, pad=1.0)
    else:
        cax.axis("off")


def draw_overview_page(pdf: PdfPages, result: ConversionResult, page_groups: Sequence[ChannelGroup], page_number: int, total_pages: int) -> None:
    """Draw one compact PDF overview page with fixed square panels."""
    fig = plt.figure(figsize=(PDF_FIG_WIDTH, PDF_FIG_HEIGHT))
    fig.patch.set_facecolor("white")

    fig.suptitle(
        f"{result.nhf_path.name} | channel groups page {page_number}/{total_pages} | output: {result.gwy_path.name if result.gwy_path else 'not created'}",
        fontsize=10.0,
        y=0.985,
    )
    left_margin = 0.022
    right_margin = 0.030
    bottom_margin = 0.032
    top_margin = 0.066
    hgap = 0.020
    vgap = 0.044

    cell_width = (1.0 - left_margin - right_margin - hgap) / PDF_GRID_NCOLS
    cell_height = (1.0 - bottom_margin - top_margin - vgap) / PDF_GRID_NROWS

    for idx in range(PDF_GRID_NROWS * PDF_GRID_NCOLS):
        row = idx // PDF_GRID_NCOLS
        col = idx % PDF_GRID_NCOLS
        cell_left = left_margin + col * (cell_width + hgap)
        cell_bottom = bottom_margin + (PDF_GRID_NROWS - 1 - row) * (cell_height + vgap)
        cell = (cell_left, cell_bottom, cell_width, cell_height)

        if idx < len(page_groups):
            draw_channel_group_at(fig, cell, page_groups[idx])

    pdf.savefig(fig, dpi=170)
    plt.close(fig)

def save_folder_overview_pdf(folder: Path, results: Sequence[ConversionResult]) -> Optional[Path]:
    """Create one overview PDF in a folder for the NHF files directly inside it."""
    valid_results = [result for result in results if result.channels]
    if not valid_results:
        return None

    pdf_path = folder / "nhf_overview.pdf"
    if pdf_path.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping existing PDF: {pdf_path}")
        return pdf_path

    with PdfPages(pdf_path) as pdf:
        for result in valid_results:
            groups = build_channel_groups(result.channels)
            chunks = [
                groups[i : i + CHANNEL_GROUPS_PER_PDF_PAGE]
                for i in range(0, len(groups), CHANNEL_GROUPS_PER_PDF_PAGE)
            ]
            for page_index, chunk in enumerate(chunks, start=1):
                draw_overview_page(pdf, result, chunk, page_index, len(chunks))

    return pdf_path


def convert_one_file(nhf_path: Path) -> ConversionResult:
    """Convert one NHF file to one GWY file."""
    print(f"\nReading: {nhf_path}")
    try:
        channels, metadata = read_nhf_channels(nhf_path)
        if not channels:
            return ConversionResult(nhf_path, None, [], metadata, "No image-like datasets found.")

        output_path = nhf_path.with_suffix(".gwy")
        if output_path.exists() and not OVERWRITE_EXISTING:
            print(f"Skipping existing GWY: {output_path}")
        else:
            write_gwy_file(channels, output_path)
            print(f"Saved GWY: {output_path}")

        print(f"Extracted {len(channels)} channel(s).")
        return ConversionResult(nhf_path, output_path, channels, metadata)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR for {nhf_path}: {error}")
        return ConversionResult(nhf_path, None, [], {}, error)


def collect_nhf_files(input_path: Path) -> List[Path]:
    """Collect one file or all .nhf files recursively from a folder."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".nhf" else []

    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".nhf"
    )


def ask_input_path() -> Path:
    """Ask the user to confirm or replace the input path before processing."""
    cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_INPUT_PATH

    print("\nNHF to GWY converter")
    print("--------------------")
    print(f"Current path: {cli_path}")
    answer = input("Press Enter to use this path, or paste another .nhf file/folder path: ").strip()

    raw_path = answer if answer else str(cli_path)
    raw_path = raw_path.strip().strip('"').strip("'")
    return Path(raw_path).expanduser().resolve()


def group_results_by_folder(results: Iterable[ConversionResult]) -> Dict[Path, List[ConversionResult]]:
    """Group conversion results by the folder containing each original NHF file."""
    grouped: Dict[Path, List[ConversionResult]] = {}
    for result in results:
        grouped.setdefault(result.nhf_path.parent, []).append(result)
    return grouped


def print_summary(results: Sequence[ConversionResult], pdf_paths: Sequence[Path]) -> None:
    """Print a short conversion summary."""
    ok = [result for result in results if result.gwy_path and result.channels]
    failed = [result for result in results if result.error]

    print("\nSummary")
    print("-------")
    print(f"Converted files: {len(ok)}")
    print(f"Failed/skipped files: {len(failed)}")
    print(f"Overview PDFs: {len(pdf_paths)}")

    if failed:
        print("\nFailed/skipped:")
        for result in failed:
            print(f"- {result.nhf_path}: {result.error}")

    if pdf_paths:
        print("\nCreated overview PDFs:")
        for path in pdf_paths:
            print(f"- {path}")


def main() -> None:
    """Main command-line entry point."""
    if GWY_IMPORT_ERROR is not None:
        print("ERROR: The package 'gwyfile' could not be imported.")
        print("Install the required packages in your active environment with:")
        print("    python3 -m pip install h5py numpy matplotlib gwyfile")
        print(f"Original import error: {GWY_IMPORT_ERROR}")
        sys.exit(1)

    input_path = ask_input_path()
    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)

    nhf_files = collect_nhf_files(input_path)
    if not nhf_files:
        print(f"ERROR: No .nhf files found in: {input_path}")
        sys.exit(1)

    print(f"\nFound {len(nhf_files)} .nhf file(s).")
    results = [convert_one_file(path) for path in nhf_files]

    pdf_paths: List[Path] = []
    for folder, folder_results in group_results_by_folder(results).items():
        pdf_path = save_folder_overview_pdf(folder, folder_results)
        if pdf_path is not None:
            pdf_paths.append(pdf_path)
            print(f"Saved overview PDF: {pdf_path}")

    print_summary(results, pdf_paths)


if __name__ == "__main__":
    main()
