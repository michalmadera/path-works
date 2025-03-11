"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))


import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()
import pyvips
import warnings
from typing import TYPE_CHECKING, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

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
warnings.filterwarnings("ignore", message="No GPU detected or cuda not installed, torch.compile is only supported")

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda:0"
else:
    print("CUDA is not available. Using CPU instead.")
    DEVICE = "cpu"


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
        units="power",
        patch_input_shape=[1024, 1024],
        patch_output_shape=[512, 512],
        stride_shape=[128, 128],
        crash_on_exception=True,
        device=DEVICE  # Use device parameter instead of on_gpu
    )
    return output

def process_region(svs_path: str, location: list[int], size: list[int], level: int = 0) -> np.ndarray:
    wsi_reader = WSIReader.open(input_img=svs_path)
    sample = wsi_reader.read_region(
        location=location,
        level=level,
        size=size,
    )

    if sample.shape[2] == 4:
        sample = cv2.cvtColor(sample, cv2.COLOR_BGRA2BGR)

    return sample

def make_prediction(svs_path: str, location: list[int], size: list[int], save_dir: str, level: int = 0) -> Dict[str, np.ndarray]:

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        sample = process_region(svs_path=svs_path, location=location, level=level, size=size)

        temp_sample_path = os.path.join(temp_dir, "temp_sample.tif")
        temp_img = pyvips.Image.new_from_memory(sample.data, sample.shape[1], sample.shape[0], sample.shape[2], 'uchar')
        temp_img.write_to_file(temp_sample_path, tile=False, compression="jpeg", Q=70)

        output_save_dir = os.path.join(temp_dir, "output")
        output_list = semantic_seg_engine(
            wsi_paths=[temp_sample_path],
            msk_paths=None,
            save_dir=output_save_dir,
            mode="tile"
        )

        logger.info(
            "Prediction method output is: %s, %s",
            output_list[0][0],
            output_list[0][1],
        )
        wsi_prediction_raw = np.load(
            output_list[0][1] + ".raw.0.npy",
        )

        wsi_prediction = np.argmax(
            wsi_prediction_raw,
            axis=-1,
        )

        class_masks = {}
        labels = ["Tumour", "Stroma", "Inflamatory", "Necrosis", "Others"]
        for i, label in enumerate(labels):
            class_masks[label] = (wsi_prediction == i).astype(np.uint8) * 255

    return class_masks