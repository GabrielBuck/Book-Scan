"""Etapa 8: transformações de intensidade aplicadas só à luminância.

Trabalhar no canal L do espaço LAB (e não em R, G e B separadamente) muda o brilho
e o contraste sem alterar a proporção entre as cores, preservando o tom da capa.
"""
import cv2
import numpy as np

METHODS = {
    "clahe": "CLAHE (equalização adaptativa limitada)",
    "equalize": "Equalização global de histograma",
    "stretch": "Alongamento linear de contraste",
    "gamma": "Correção gama",
    "none": "Sem tratamento",
}


def _apply_to_luminance(image_bgr: np.ndarray, func) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    merged = cv2.merge((func(l_channel), a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def gamma_lut(gamma: float) -> np.ndarray:
    """Tabela s = 255·(r/255)^γ: γ<1 clareia, γ>1 escurece (formulação de Gonzalez & Woods)."""
    r = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(255.0 * np.power(r, gamma), 0, 255).astype(np.uint8)


def contrast_stretch(channel: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Transformação linear que leva os percentis [1, 99] para [0, 255].
    Os percentis (em vez de min/max) impedem que pixels isolados anulem o efeito."""
    low, high = np.percentile(channel, (low_pct, high_pct))
    if high - low < 10:  # imagem praticamente uniforme: esticar só amplificaria ruído
        return channel
    out = (channel.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance(
    image_bgr: np.ndarray,
    method: str = "clahe",
    strength: float = 1.0,
    clip_limit: float = 2.0,
    tile_size: int = 8,
    gamma: float = 1.0,
):
    """Aplica o tratamento escolhido. Retorna (imagem_tratada, descrição legível).

    `strength` (0–1) mistura o resultado com a imagem original para evitar um
    aspecto artificial.
    """
    if method == "clahe":
        # CLAHE equaliza por blocos e corta o histograma em `clip_limit`, o que
        # evita estourar o contraste de regiões homogêneas (e amplificar ruído)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size))
        treated = _apply_to_luminance(image_bgr, clahe.apply)
        description = f"CLAHE no canal L (LAB), clipLimit={clip_limit:g}, blocos {tile_size}×{tile_size}"
    elif method == "equalize":
        treated = _apply_to_luminance(image_bgr, cv2.equalizeHist)
        description = "Equalização de histograma global no canal L (LAB)"
    elif method == "stretch":
        treated = _apply_to_luminance(image_bgr, contrast_stretch)
        description = "Alongamento linear de contraste no canal L (percentis 1–99 → 0–255)"
    elif method == "gamma":
        lut = gamma_lut(gamma)
        treated = _apply_to_luminance(image_bgr, lambda ch: cv2.LUT(ch, lut))
        description = f"Correção gama no canal L (γ = {gamma:g})"
    else:
        return image_bgr.copy(), "Nenhum tratamento aplicado"

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1.0:
        treated = cv2.addWeighted(treated, strength, image_bgr, 1.0 - strength, 0)
        description += f", intensidade {strength:.0%}"
    return treated, description
