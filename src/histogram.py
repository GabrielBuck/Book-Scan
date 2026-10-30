"""Etapa 9: histogramas de luminância antes e depois do tratamento."""
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def luminance(image_bgr: np.ndarray) -> np.ndarray:
    """Canal L (0–255) do espaço LAB: o mesmo canal que o tratamento modifica."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """h(r_k) = número de pixels com intensidade r_k, para k = 0..255."""
    return cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()


def histogram_stats(image_bgr: np.ndarray) -> dict:
    """Métricas simples da luminância para comparar antes/depois."""
    l_channel = luminance(image_bgr)
    p1, p99 = np.percentile(l_channel, (1, 99))
    return {
        "média": float(l_channel.mean()),
        "desvio padrão": float(l_channel.std()),  # medida usual de contraste global
        "faixa dinâmica (p1–p99)": float(p99 - p1),
    }


def _draw_panel(ax, image_bgr: np.ndarray, title: str, y_max: float) -> None:
    l_hist = compute_histogram(luminance(image_bgr))
    x = np.arange(256)
    ax.fill_between(x, l_hist, color="#7a7a7a", alpha=0.55, label="Luminância (L)")
    for idx, (name, color) in enumerate((("B", "#2d6fd6"), ("G", "#2e9e4f"), ("R", "#d63a3a"))):
        h = cv2.calcHist([image_bgr], [idx], None, [256], [0, 256]).ravel()
        ax.plot(x, h, color=color, linewidth=0.8, alpha=0.8, label=name)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Intensidade (0–255)", fontsize=9)
    ax.tick_params(labelsize=8)

    # a distribuição acumulada mostra por que a equalização "espalha" as intensidades
    cdf = np.cumsum(l_hist) / max(l_hist.sum(), 1)
    twin = ax.twinx()
    twin.plot(x, cdf, color="black", linestyle="--", linewidth=1, label="Acumulado (L)")
    twin.set_ylim(0, 1.02)
    twin.tick_params(labelsize=8)


def render_histograms(before_bgr: np.ndarray, after_bgr: np.ndarray, after_label: str) -> np.ndarray:
    """Figura com os histogramas lado a lado, na mesma escala vertical. Retorna imagem BGR."""
    fig = Figure(figsize=(10, 3.4), dpi=110)
    canvas = FigureCanvasAgg(fig)
    ax_before, ax_after = fig.subplots(1, 2)

    y_max = 1.05 * max(
        compute_histogram(luminance(img)).max() for img in (before_bgr, after_bgr)
    )
    _draw_panel(ax_before, before_bgr, "Antes do tratamento", y_max)
    _draw_panel(ax_after, after_bgr, f"Depois: {after_label}", y_max)
    ax_before.set_ylabel("Número de pixels", fontsize=9)
    ax_before.legend(fontsize=7, loc="upper left")
    fig.tight_layout()

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(rgba[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
