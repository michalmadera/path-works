"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
import pyvips
import os
import warnings
from typing import TYPE_CHECKING, Callable

# Third party imports

import matplotlib.pyplot as plt
import numpy as np



from tiatoolbox.data import stain_norm_target


from tiatoolbox.tools.stainnorm import get_normalizer

from tiatoolbox.wsicore.wsireader import WSIReader

from tiatoolbox import logger
from tiatoolbox import utils

if TYPE_CHECKING:  # pragma: no cover

    from numpy.typing import ArrayLike
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
import cv2

warnings.filterwarnings("ignore")


def semantic_seg_engine(
        wsi_paths: list[str],
        msk_paths: list[str],
        save_dir: str,
        mode: str,
        on_gpu,
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
        on_gpu=on_gpu,
        resolution=1.0,
        units="baseline",
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[128, 128],
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

def overlay_prediction_mask(img, prediction, alpha=0.35, label_info=None, min_val=0.0, excluded_classes=None, ax=None, return_ax=False):
    if img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)  # ensure image is RGBA

    # Prepare the overlay image with an alpha channel
    overlay = np.ones((img.shape[0], img.shape[1], 4), dtype=np.uint8) * 255  # white background in RGBA

    # Prepare the RGB prediction mask
    rgb_prediction = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    # Apply colors based on the prediction mask
    for label_uid, (_, overlay_rgb) in (label_info or {}).items():
        if excluded_classes and label_uid in excluded_classes:
            continue  # Skip excluded classes
        mask = prediction == label_uid
        rgb_prediction[mask] = np.array(overlay_rgb, dtype=np.uint8)

    # Perform weighted addition
    result_rgb = cv2.addWeighted(rgb_prediction, alpha, overlay[:, :, :3], 1 - alpha, 0)

    # Combine result with the alpha channel
    overlay[:, :, :3] = result_rgb
    overlay[:, :, 3] = 255  # Fully opaque by default

    # Set alpha to 0 where classes are to be excluded (transparent areas)
    if excluded_classes:
        for class_id in excluded_classes:
            overlay[prediction == class_id, 3] = 0  # make excluded classes fully transparent

    if ax is None and not return_ax:
        return overlay

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB))
    ax.axis("off")
    return ax if return_ax else overlay


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

def make_prediction(svs_path: str, location: list[int], size: list[int], save_path: str, save_dir: str, level: int = 0, on_gpu = True) -> np.ndarray:
    sample_path = save_svs_region(svs_path=svs_path, location=location, level=level, size=size, save_path=save_path)
    output_save_dir = os.path.join(save_dir, "output")
    output_list = semantic_seg_engine(
        wsi_paths=[sample_path],
        msk_paths=None,
        save_dir=output_save_dir,
        mode="tile",
        on_gpu=on_gpu
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
        (1, 0.5, 0), #Pomarańczowy
        (0, 1, 1),  # Cyjan
        (1, 1, 1)  # Biały
    ]
    for class_name, label in label_dict.items():
        label_color_dict[label] = (class_name, 255 * np.array(colors[label]))
    # Creat overlay map using the `overlay_prediction_mask` helper function
    overlay = overlay_prediction_mask(
        img=wsi_overview,
        prediction=wsi_prediction,
        alpha=0.3,
        label_info=label_color_dict,
        excluded_classes=[1, 2, 3, 4],  # Lista identyfikatorów klas do wykluczenia
        return_ax=False
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


    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    new_overlay = np.zeros((*overlay.shape[:2], 4), dtype=np.uint8)  # Create a new RGBA overlay
    new_overlay[:, :, :3] = overlay[:, :, :3]  # Copy the RGB channels
    new_overlay[:, :, 3] = overlay[:, :, 3]    # Copy the alpha channel

    return new_overlay
def overlay_tif_with_pred(svs_path: str, overlay: np.ndarray, save_path: str, location: list[int]):
    # Open the main image using WSIReader and get the thumbnail
    wsi_reader = WSIReader.open(input_img=svs_path)
    img1 = wsi_reader.slide_thumbnail(resolution=0, units="level")

    # Ensure both images are in RGBA to handle transparency
    if img1.shape[2] == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)

    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # Define position and boundaries
    pos_x, pos_y = location
    x_end = min(pos_x + overlay.shape[1], img1.shape[1])
    y_end = min(pos_y + overlay.shape[0], img1.shape[0])

    # Alpha blending factor
    alpha = 0.5
    # Apply the overlay on the base image
    sub_img = img1[pos_y:y_end, pos_x:x_end]
    sub_overlay = overlay[:y_end-pos_y, :x_end-pos_x]
    alpha_overlay = sub_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay * alpha  # adjusted alpha for background based on overlay alpha

    # Blend the overlay and the image
    for c in range(3):  # RGB channels
        sub_img[:, :, c] = (alpha_background * sub_img[:, :, c] + alpha_overlay * alpha * sub_overlay[:, :, c])

    # Convert back to RGB for saving if needed
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    img1_vips = pyvips.Image.new_from_memory(img1_rgb.data, img1_rgb.shape[1], img1_rgb.shape[0], img1_rgb.shape[2], 'uchar')

    # Define the full path and save using PyVIPS
    save_full_path = os.path.join(save_path, "result.tif")
    img1_vips.write_to_file(save_full_path, tile=True, compression="jpeg")

    return save_full_path