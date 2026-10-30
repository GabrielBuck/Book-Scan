"""Etapa 4: detecção de bordas (Canny) e fechamento de lacunas."""
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import gaussian_smooth

DEFAULT_LOW, DEFAULT_HIGH = 20, 50  # limiares baixos: a validação geométrica descarta o excesso de bordas


@dataclass
class EdgeResult:
    edges: np.ndarray         # mapa de bordas (exibido ao usuário)
    edges_closed: np.ndarray  # bordas com lacunas fechadas (entrada da busca de contornos)
    low: int
    high: int


def detect_edges(blurred_gray, image_bgr, ksize=5, low=None, high=None) -> EdgeResult:
    """Canny: gradiente (Sobel) -> supressão de não-máximos -> histerese com dois limiares.

    O Canny é aplicado à imagem em cinza suavizada e também a cada canal B, G e R
    (suavizados do mesmo modo), unindo os resultados. Duas cores diferentes podem ter o
    mesmo brilho em cinza (ex.: capa vermelha sobre mesa cinza); nesse caso a borda só
    aparece em algum dos canais.
    """
    low = DEFAULT_LOW if low is None else int(low)
    high = DEFAULT_HIGH if high is None else int(high)
    high = max(high, low + 1)

    edges = cv2.Canny(blurred_gray, low, high)
    for channel in cv2.split(image_bgr):
        edges |= cv2.Canny(gaussian_smooth(channel, ksize), low, high)

    # Bordas reais quase sempre têm pequenas falhas (reflexos, dobras, sombra). Uma
    # dilatação seguida de fechamento morfológico as religa para formar contornos fechados.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel, iterations=2)
    return EdgeResult(edges=edges, edges_closed=closed, low=low, high=high)
