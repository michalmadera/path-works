from __future__ import annotations

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from tiatoolbox import logger
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader


def convert_float_rgb_to_uint8(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Konwertuje kolor RGB z wartości zmiennoprzecinkowych (0-1) na uint8 (0-255)."""
    return tuple(int(c * 255) for c in color)


def save_region_as_png(
        svs_path: str,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        save_path: str
) -> str:
    """Zapisuje region WSI jako plik PNG."""
    wsi_reader = WSIReader.open(input_img=svs_path)
    sample = wsi_reader.read_region(
        location=tuple(location),
        level=level,
        size=tuple(size),
    )

    if sample.shape[2] == 4:  # Konwersja BGRA do BGR
        sample = cv2.cvtColor(sample, cv2.COLOR_BGRA2BGR)

    # Zapisz jako PNG
    save_full_path = save_path.replace('.svs', '.png').replace('.tif', '.png')
    cv2.imwrite(save_full_path, sample)

    return save_full_path


def make_prediction(
        svs_path: str,
        location: Tuple[int, int],
        size: Tuple[int, int],
        save_path: str,
        save_dir: str,
        level: int = 0,
        on_gpu: bool = True
) -> np.ndarray:
    """Wykonuje predykcję segmentacji na wybranym regionie WSI."""
    # Przygotowanie ścieżek
    sample_path = save_region_as_png(svs_path, location, level, size, save_path)
    output_save_dir = Path(save_dir) / "output"
    output_save_dir.mkdir(parents=True, exist_ok=True)

    # Inicjalizacja modelu segmentacji
    bcc_segmentor = SemanticSegmentor(
        pretrained_model="fcn_resnet50_unet-bcss",
        num_loader_workers=2,
        batch_size=16,
        auto_generate_mask=True,
        verbose=True
    )

    # Predykcja
    output = bcc_segmentor.predict(
        imgs=[sample_path],
        masks=None,
        mode="tile",
        patch_input_shape=(1024, 1024),
        patch_output_shape=(512, 512),
        stride_shape=(128, 128),
        resolution=1.0,
        units="power",
        save_dir=str(output_save_dir),
        device="cuda" if on_gpu else "cpu",
        crash_on_exception=True
    )

    # Wczytanie predykcji
    wsi_prediction_raw = np.load(output[0][1] / ".raw.0.npy")
    prediction_mask = np.argmax(wsi_prediction_raw, axis=-1)

    # Definicja kolorów
    colors = [
        (1, 1, 0),  # Żółty
        (0, 1, 0),  # Zielony
        (1, 0.5, 0),  # Pomarańczowy
        (0, 1, 1),  # Cyjan
        (1, 1, 1)  # Biały
    ]

    # Przygotowanie słownika kolorów
    label_colors = {
        i: (name, convert_float_rgb_to_uint8(color))
        for i, (name, color) in enumerate(zip(
            ["Tumour", "Stroma", "Inflamatory", "Necrosis", "Others"],
            colors
        ))
    }

    # Wczytanie oryginalnego obrazu
    wsi_reader = WSIReader.open(input_img=sample_path)
    image = wsi_reader.slide_thumbnail(resolution=1, units="power")

    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Tworzenie nakładki
    overlay = create_prediction_overlay(
        image=image,
        prediction=prediction_mask,
        label_colors=label_colors,
        excluded_classes=[1, 2, 3, 4]
    )

    return overlay


def create_prediction_overlay(
        image: np.ndarray,
        prediction: np.ndarray,
        label_colors: Dict[int, Tuple[str, Tuple[int, int, int]]],
        excluded_classes: Optional[List[int]] = None,
        alpha: float = 0.3
) -> np.ndarray:
    """Tworzy nakładkę predykcji na obrazie."""
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    for label_id, (_, color) in label_colors.items():
        if excluded_classes and label_id in excluded_classes:
            continue

        mask = prediction == label_id
        overlay[mask] = (*color, 255)

    overlay[..., 3] = (overlay[..., 3] * alpha).astype(np.uint8)

    return overlay


def overlay_png_with_pred(
        svs_path: str,
        overlay: np.ndarray,
        save_path: str,
        location: Tuple[int, int]
) -> str:
    """Nakłada predykcję na oryginalny obraz i zapisuje jako PNG."""
    wsi_reader = WSIReader.open(input_img=svs_path)
    img = wsi_reader.slide_thumbnail(resolution=0, units="level")

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    pos_x, pos_y = location
    x_end = min(pos_x + overlay.shape[1], img.shape[1])
    y_end = min(pos_y + overlay.shape[0], img.shape[0])

    sub_img = img[pos_y:y_end, pos_x:x_end]
    sub_overlay = overlay[:y_end - pos_y, :x_end - pos_x]

    alpha_overlay = sub_overlay[..., 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        sub_img[..., c] = (alpha_background * sub_img[..., c] +
                           alpha_overlay * sub_overlay[..., c])

    save_full_path = str(Path(save_path) / "result.png")
    cv2.imwrite(save_full_path, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

    return save_full_path
