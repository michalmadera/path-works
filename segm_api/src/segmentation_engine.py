"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
import pyvips
import copy
import os
import random
import shutil
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third party imports
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

# Use ujson as replacement for default json because it's faster for large JSON
import ujson as json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from tiatoolbox import logger
from tiatoolbox.data import stain_norm_target
from tiatoolbox.models import (
    IOSegmentorConfig,

)

from tiatoolbox.tools.stainnorm import get_normalizer

from tiatoolbox.wsicore.wsireader import WSIReader

from tiatoolbox import logger
from tiatoolbox import utils
from tiatoolbox.tools import stainnorm
if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator
    from numpy.typing import ArrayLike
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
import cv2
from matplotlib import cm
warnings.filterwarnings("ignore")


def semantic_seg_engine(
        wsi_paths: list[str],
        msk_paths: list[str],
        save_dir: str,
        mode: str,
        preproc_func: Callable | None = None,
) -> list:
    bcc_segmentor = SemanticSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        num_loader_workers=2,
        batch_size=16,
        auto_generate_mask=True,
        verbose=True
    )

    bcc_segmentor.model.preproc_func = preproc_func

    output = bcc_segmentor.predict(
        wsi_paths,
        msk_paths,
        save_dir=save_dir,
        mode=mode,
        resolution=1.0,
        units="baseline",
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[128, 128],
        on_gpu=True,
        crash_on_exception=True,
    )
    return output




def stain_norm_func(img: np.ndarray) -> np.ndarray:
    target_image = stain_norm_target()
    stain_normalizer = get_normalizer("macenko")
    stain_normalizer.fit(target_image)
    """Helper function to perform stain normalization."""
    return stain_normalizer.transform(img)


def save_svs_region(svs_path: str, location: list[int], level: int, size: list[int], save_path: str) -> str:
    wsi_reader = WSIReader.open(input_img=svs_path)
    sample = wsi_reader.read_region(
        location=location,
        level=level,
        size=size,
    )
    utils.misc.imwrite(save_path, sample)
    return save_path

def overlay_prediction_mask(
    img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.35,
    label_info: dict | None = None,
    min_val: float = 0.0,
    excluded_class: int | None = None,
    ax: mpl.axes.Axes | None = None,
    *,
    return_ax: bool,
) -> np.ndarray | mpl.axes.Axes:
    """Generate an overlay, given a 2D prediction map.

    Args:
        ...
        excluded_class (int | None):
            Identyfikator klasy do wykluczenia z mapy nakładki.
        ...

    Returns:
        If return_ax is True, return the matplotlib ax object. Else,
        return the overlay array.

    """
    # Validate inputs
    if img.shape[:2] != prediction.shape[:2]:
        msg = (
            f"Mismatch shape `img` {img.shape[:2]} "
            f"vs `prediction` {prediction.shape[:2]}."
        )
        raise ValueError(
            msg,
        )
    if np.issubdtype(img.dtype, np.floating):
        if not (img.max() <= 1.0 and img.min() >= 0):
            msg = "Not support float `img` outside [0, 1]."
            raise ValueError(msg)
        img = np.array(img * 255, dtype=np.uint8)
    # If `min_val` is defined, only display the overlay for areas with pred > min_val
    if min_val > 0:
        prediction_sel = prediction >= min_val

    overlay = np.ones_like(img) * 255  # Set white background

    predicted_classes = sorted(np.unique(prediction).tolist())
    rand_state = np.random.default_rng().__getstate__()
    rng = np.random.default_rng(123)
    label_info = label_info or {  # Use label_info if provided OR generate
        label_uid: (str(label_uid), rng.integers(0, 255, 3))
        for label_uid in predicted_classes
    }
    np.random.default_rng().__setstate__(rand_state)

    missing_label_uids = _validate_label_info(label_info, predicted_classes)
    if len(missing_label_uids) != 0:
        msg = f"Missing label for: {missing_label_uids}."
        raise ValueError(msg)

    rgb_prediction = np.zeros(
        [prediction.shape[0], prediction.shape[1], 3],
        dtype=np.uint8,
    )
    for label_uid, (_, overlay_rgb) in label_info.items():
        if label_uid == excluded_class:  # If excluded class, set background to white
            overlay_rgb = [255, 255, 255]
        sel = prediction == label_uid
        rgb_prediction[sel] = overlay_rgb

    cv2.addWeighted(rgb_prediction, alpha, overlay, 1 - alpha, 0, overlay)
    overlay = overlay.astype(np.uint8)

    if min_val > 0.0:
        overlay[~prediction_sel] = img[~prediction_sel]

    if ax is None and not return_ax:
        return overlay

    name_list, color_list = zip(*label_info.values())
    color_list = np.array(color_list) / 255
    uid_list = list(label_info.keys())
    cmap = mpl.colors.ListedColormap(color_list)

    colorbar_params = {
        "mappable": mpl.cm.ScalarMappable(cmap=cmap),
        "boundaries": [*uid_list, uid_list[-1] + 1],
        "values": uid_list,
        "ticks": [b + 0.5 for b in uid_list],
        "spacing": "proportional",
        "orientation": "vertical",
    }

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(overlay)
    ax.axis("off")
    cbar = plt.colorbar(**colorbar_params, ax=ax)
    cbar.ax.set_yticklabels(name_list)
    cbar.ax.tick_params(labelsize=12)

    return ax

def _validate_label_info(
    label_info: dict[int, tuple[str, ArrayLike]],
    predicted_classes: list,
) -> list[int]:
    """Validate the label_info dictionary.

    Args:
        label_info (dict):
            A dictionary containing the mapping for each integer value
            within `prediction` to its string and color. [int] : (str,
            (int, int, int)).
        predicted_classes (list):
            List of predicted classes.

    Raises:
        ValueError:
            If the label_info dictionary is not valid.

    Returns:
        list:
            List of missing label UIDs.

    """
    # May need better error messages
    check_uid_list = predicted_classes.copy()
    for label_uid, (label_name, label_colour) in label_info.items():
        if label_uid in check_uid_list:
            check_uid_list.remove(label_uid)
        if not isinstance(label_uid, int):
            msg = (
                f"Wrong `label_info` format: label_uid "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if not isinstance(label_name, str):
            msg = (
                f"Wrong `label_info` format: label_name "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if not isinstance(label_colour, (tuple, list, np.ndarray)):
            msg = (
                f"Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise TypeError(
                msg,
            )
        if len(label_colour) != 3:  # noqa: PLR2004
            msg = (
                f"Wrong `label_info` format: label_colour "
                f"{[label_uid, (label_name, label_colour)]}"
            )
            raise ValueError(
                msg,
            )

    return check_uid_list

def make_prediction(svs_path: str, location: list[int], size: list[int], save_path: str, save_dir: str, level: int = 0) -> np.ndarray:
    sample_path = save_svs_region(svs_path=svs_path, location=location, level=level, size=size, save_path=save_path)
    output_save_dir = os.path.join(save_dir, "output")
    output_list = semantic_seg_engine(
        wsi_paths=[sample_path],
        msk_paths=None,
        save_dir=output_save_dir,
        mode="tile",
        #preproc_func tylko jeżeli mamy pewność że zdjęcie nie zostało wcześniej znormalizowane
        #preproc_func=stain_norm_func
    )

    bcc_wsi_ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 0.25}],
        output_resolutions=[{"units": "mpp", "resolution": 0.25}],
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        save_resolution={"units": "mpp", "resolution": 2},
    )
    logger.info(
        "Prediction method output is: %s, %s",
        output_list[0][0],
        output_list[0][1],
    )
    wsi_prediction_raw = np.load(
        output_list[0][1] + ".raw.0.npy",
    )  # Loading the first prediction [0] based on the output address [1]
    logger.info(
        "Raw prediction dimensions: (%d, %d, %d)",
        wsi_prediction_raw.shape[0],
        wsi_prediction_raw.shape[1],
        wsi_prediction_raw.shape[2],
    )

    # Simple processing of the raw prediction to generate semantic segmentation task
    wsi_prediction = np.argmax(
        wsi_prediction_raw,
        axis=-1,
    )  # select the class with highest probability
    logger.info(
        "Processed prediction dimensions: (%d, %d)",
        wsi_prediction.shape[0],
        wsi_prediction.shape[1],
    )

    # [WSI overview extraction]
    # Now reading the WSI to extract it's overview
    wsi = WSIReader.open(output_list[0][0], mpp=2)
    logger.info(
        "WSI original dimensions: (%d, %d)",
        wsi.info.slide_dimensions[0],
        wsi.info.slide_dimensions[1],
    )

    # using the prediction save_resolution to create the wsi overview at the same resolution
    overview_info = bcc_wsi_ioconfig.save_resolution

    # extracting slide overview using `slide_thumbnail` method
    wsi_overview = wsi.slide_thumbnail(
        resolution=overview_info["resolution"],
        units=overview_info["units"],
    )
    logger.info(
        "WSI overview dimensions: (%d, %d)",
        wsi_overview.shape[0],
        wsi_overview.shape[1],
    )

    # plt.figure(), plt.imshow(wsi_overview)
    # plt.axis("off")

    # [Overlay map creation]
    # creating label-color dictionary to be fed into `overlay_prediction_mask` function
    # to help generating color legend
    label_dict = {"Tumour": 0, "Stroma": 1, "Inflamatory": 2, "Necrosis": 3, "Others": 4}
    label_color_dict = {}
    colors = [
        (0, 1, 0),  # Zielony
        (1, 1, 0),  # Żółty
        (1, 0, 1),  # Magenta
        (0, 1, 1),  # Cyjan
        (1, 1, 1)  # Biały
    ]
    for class_name, label in label_dict.items():
        label_color_dict[label] = (class_name, 255 * np.array(colors[label]))
    # Creat overlay map using the `overlay_prediction_mask` helper function
    overlay = overlay_prediction_mask(
        wsi_overview,
        wsi_prediction,
        alpha=0.3,
        label_info=label_color_dict,
        excluded_class=4,
        return_ax=False,
    )

    #to zostawiam jakby trzeba było zrobić legendę
    # legend_elements = []
    # label_dict = {"Tumour": 0, "Stroma": 1, "Inflamatory": 2, "Necrosis": 3}
    # colors = [
    #     (0, 1, 0),  # Zielony
    #     (1, 1, 0),  # Żółty
    #     (1, 0, 1),  # Magenta
    #     (0, 1, 1),  # Cyjan
    #     (1, 1, 1)  # Biały
    # ]
    # for class_name, label in label_dict.items():
    #     color = 255 * np.array(colors[label])
    #     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
    #                                       markerfacecolor=color / 255, markersize=10))


    new_overlay = np.zeros((*overlay.shape[:2], 4), dtype=np.uint8)
    new_overlay[:, :, :3] = overlay
    new_overlay[:, :, 3] = 255
    white_pixels = np.all(new_overlay[:, :, :3] == [255, 255, 255], axis=-1)
    new_overlay[white_pixels, 3] = 0

    return new_overlay

def overlay_tif_with_pred(svs_path: str, overlay: np.ndarray, save_path: str, location: list[int]):
    wsi_reader = WSIReader.open(input_img=svs_path)

    img1 = wsi_reader.slide_thumbnail(resolution=0, units="level")
    save_path = os.path.join(save_path, "result.tif")
    img2 = overlay
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2RGB)
    pos_x, pos_y = location[0],  location[1]
    x_end = pos_x + img2.shape[1]
    y_end = pos_y + img2.shape[0]

    alpha = 0.5
    img1_mod = img1.copy()
    img1_mod[pos_y:y_end, pos_x:x_end] = img1[pos_y:y_end, pos_x:x_end] * alpha + img2 * (1 - alpha)

    img1_mod_rgb = cv2.cvtColor(img1_mod, cv2.COLOR_BGR2RGB)
    img1_mod_vips = pyvips.Image.new_from_memory(img1_mod_rgb.data, img1_mod_rgb.shape[1], img1_mod_rgb.shape[0],
                                                 img1_mod_rgb.shape[2], 'uchar')

    img1_mod_vips.write_to_file(save_path, tile=True, compression="jpeg")
