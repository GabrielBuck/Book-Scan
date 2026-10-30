"""Etapa 10: padronização do tamanho com interpolação."""
import cv2
import numpy as np

DEFAULT_SIZE = (400, 600)  # (largura, altura): capa vertical na proporção 2:3
SIZE_LIMITS = (50, 4000)

INTERPOLATIONS = {
    "auto": ("Automática", None),
    "area": ("INTER_AREA (média por área)", cv2.INTER_AREA),
    "linear": ("INTER_LINEAR (bilinear)", cv2.INTER_LINEAR),
    "cubic": ("INTER_CUBIC (bicúbica)", cv2.INTER_CUBIC),
    "lanczos": ("INTER_LANCZOS4 (Lanczos 8×8)", cv2.INTER_LANCZOS4),
    "nearest": ("INTER_NEAREST (vizinho mais próximo)", cv2.INTER_NEAREST),
}

_REASONS = {
    "area": "reduz a imagem calculando a média dos pixels cobertos, o que evita serrilhado (aliasing)",
    "linear": "interpola linearmente entre os 4 vizinhos mais próximos; rápida, com resultado suave",
    "cubic": "interpola com os 16 vizinhos (bicúbica), preservando melhor detalhes ao ampliar",
    "lanczos": "interpola com núcleo sinc janelado em 8×8 vizinhos; nítida, mas pode gerar halos",
    "nearest": "copia o vizinho mais próximo; sem suavização, produz efeito de blocos",
}


def choose_interpolation(src_size, dst_size) -> str:
    """Redução em ambos os eixos -> INTER_AREA; caso contrário (ampliação) -> INTER_CUBIC."""
    (sw, sh), (dw, dh) = src_size, dst_size
    return "area" if dw <= sw and dh <= sh else "cubic"


def resize_cover(
    image: np.ndarray,
    size=DEFAULT_SIZE,
    interpolation: str = "auto",
    fit_mode: str = "stretch",
):
    """Redimensiona a capa para `size` (largura, altura).

    fit_mode "stretch": ocupa todo o quadro (a capa já foi retificada, então a distorção é pequena).
    fit_mode "fit": preserva a proporção e completa o quadro com fundo neutro.
    Retorna (imagem, nome_da_interpolação_usada, explicação).
    """
    width, height = (int(v) for v in size)
    if not all(SIZE_LIMITS[0] <= v <= SIZE_LIMITS[1] for v in (width, height)):
        raise ValueError(f"Tamanho de saída deve estar entre {SIZE_LIMITS[0]} e {SIZE_LIMITS[1]} px.")
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Interpolação desconhecida: {interpolation}")

    src_h, src_w = image.shape[:2]
    key = choose_interpolation((src_w, src_h), (width, height)) if interpolation == "auto" else interpolation
    label, flag = INTERPOLATIONS[key]
    label = label.split(" (")[0]
    explanation = f"{label}" + (" (escolha automática)" if interpolation == "auto" else "") + f": {_REASONS[key]}."

    if fit_mode == "fit":
        scale = min(width / src_w, height / src_h)
        new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=flag)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        y0, x0 = (height - new_h) // 2, (width - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas, key, explanation

    return cv2.resize(image, (width, height), interpolation=flag), key, explanation
