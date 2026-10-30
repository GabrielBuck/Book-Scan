import cv2
import numpy as np
import pytest

from src.enhancement import enhance, gamma_lut
from src.histogram import compute_histogram, histogram_stats, luminance, render_histograms
from src.resize import choose_interpolation, resize_cover


def _low_contrast(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(100, 140, (120, 80, 3), dtype=np.uint8)


# ---- resize ----
def test_resize_stretch_output_size():
    out, key, note = resize_cover(np.zeros((900, 600, 3), np.uint8), (400, 600))
    assert out.shape == (600, 400, 3)
    assert key == "area" and "INTER_AREA" in note


def test_resize_auto_picks_cubic_when_enlarging():
    assert choose_interpolation((200, 300), (400, 600)) == "cubic"
    assert choose_interpolation((800, 1200), (400, 600)) == "area"
    assert resize_cover(np.zeros((300, 200, 3), np.uint8), (400, 600))[1] == "cubic"


def test_resize_explicit_interpolation_and_fit_mode():
    src = np.full((300, 600, 3), 10, np.uint8)  # proporção 2:1, bem diferente de 2:3
    out, key, _ = resize_cover(src, (400, 600), interpolation="linear", fit_mode="fit")
    assert out.shape == (600, 400, 3) and key == "linear"
    assert out[0, 0].tolist() == [245, 245, 245]      # faixa de preenchimento
    assert out[300, 200].tolist() == [10, 10, 10]     # conteúdo centralizado


@pytest.mark.parametrize("size", [(0, 600), (400, 10000), (-5, 5)])
def test_resize_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        resize_cover(np.zeros((100, 100, 3), np.uint8), size)


def test_resize_rejects_unknown_interpolation():
    with pytest.raises(ValueError):
        resize_cover(np.zeros((100, 100, 3), np.uint8), interpolation="magica")


# ---- enhancement ----
@pytest.mark.parametrize("method", ["clahe", "equalize", "stretch", "gamma"])
def test_enhance_keeps_shape_and_raises_contrast(method):
    img = _low_contrast()
    out, description = enhance(img, method, gamma=0.5)
    assert out.shape == img.shape and out.dtype == np.uint8 and description
    if method != "gamma":
        assert histogram_stats(out)["desvio padrão"] > histogram_stats(img)["desvio padrão"]


def test_enhance_none_and_zero_strength_return_original():
    img = _low_contrast()
    np.testing.assert_array_equal(enhance(img, "none")[0], img)
    np.testing.assert_array_equal(enhance(img, "clahe", strength=0.0)[0], img)


def test_enhance_preserves_hue_roughly():
    # tratar só a luminância não deve inverter a cor dominante
    img = np.zeros((100, 100, 3), np.uint8)
    img[..., 2] = 150
    img[..., 1] = 60
    img[:50] = (img[:50] * 0.6).astype(np.uint8)
    out, _ = enhance(img, "clahe")
    assert out[..., 2].mean() > out[..., 0].mean()


def test_gamma_lut_direction():
    assert gamma_lut(0.5)[100] > 100 > gamma_lut(2.0)[100]


# ---- histogram ----
def test_histogram_counts_all_pixels():
    img = _low_contrast()
    hist = compute_histogram(luminance(img))
    assert hist.shape == (256,) and hist.sum() == img.shape[0] * img.shape[1]


def test_render_histograms_returns_bgr_image():
    img = _low_contrast()
    out, _ = enhance(img, "clahe")
    fig = render_histograms(img, out, "CLAHE")
    assert fig.ndim == 3 and fig.shape[2] == 3 and fig.shape[1] > fig.shape[0]
