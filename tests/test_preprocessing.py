import io

import cv2
import numpy as np
import pytest
from PIL import Image

from src.preprocessing import ImageError, gaussian_smooth, load_image, resize_for_detection, to_grayscale


def _encode(image_rgb: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format=fmt)
    return buf.getvalue()


def test_load_png_and_jpeg_return_bgr():
    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[..., 0] = 200  # canal R
    for fmt in ("PNG", "JPEG"):
        image, notes = load_image(_encode(rgb, fmt))
        assert image.shape == (300, 400, 3)
        assert image[..., 2].mean() > 150 and image[..., 0].mean() < 50  # R foi para o canal 2 do BGR
        assert notes == []


@pytest.mark.parametrize("data", [b"", b"isto nao e uma imagem", b"\x89PNG\r\n\x1a\ncorrompido"])
def test_invalid_data_raises_friendly_error(data):
    with pytest.raises(ImageError):
        load_image(data)


def test_unsupported_format_rejected():
    with pytest.raises(ImageError, match="Formato"):
        load_image(_encode(np.zeros((300, 300, 3), np.uint8), "BMP"))


def test_small_image_rejected():
    with pytest.raises(ImageError, match="pequena"):
        load_image(_encode(np.zeros((100, 120, 3), np.uint8), "PNG"))


def test_large_image_is_downscaled_with_note():
    image, notes = load_image(_encode(np.zeros((400, 1200, 3), np.uint8), "PNG"), max_side=600)
    assert max(image.shape[:2]) == 600
    assert len(notes) == 1


def test_transparent_png_composited_on_white():
    rgba = np.zeros((300, 300, 4), np.uint8)  # totalmente transparente
    image, _ = load_image(_encode(rgba, "PNG"))
    assert image.min() == 255


def test_resize_for_detection_scale():
    small, scale = resize_for_detection(np.zeros((2000, 1000, 3), np.uint8), max_side=1000)
    assert small.shape[:2] == (1000, 500) and scale == 0.5
    same, scale = resize_for_detection(np.zeros((300, 200, 3), np.uint8))
    assert scale == 1.0 and same.shape[:2] == (300, 200)


def test_grayscale_and_blur_shapes():
    bgr = np.random.default_rng(0).integers(0, 255, (50, 60, 3), dtype=np.uint8)
    gray = to_grayscale(bgr)
    assert gray.shape == (50, 60)
    noisy = gray.copy()
    assert gaussian_smooth(noisy, 4).std() < noisy.std()  # núcleo par vira ímpar e suaviza
