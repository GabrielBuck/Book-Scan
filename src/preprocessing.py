"""Etapas 1 a 3 do pipeline: leitura da imagem, escala de cinza e filtragem espacial."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = ("JPEG", "PNG")
MIN_SIDE = 200          # lado menor abaixo disso não tem informação suficiente para detectar a capa
MAX_INPUT_SIDE = 3000   # fotos de celular maiores que isso são reduzidas (memória e tempo)
DETECTION_SIDE = 1000   # a detecção roda em uma cópia reduzida; a correção usa a imagem completa


class ImageError(ValueError):
    """Problema com a imagem enviada; a mensagem pode ser exibida ao usuário."""


def load_image(data: bytes, max_side: int = MAX_INPUT_SIDE, min_side: int = MIN_SIDE):
    """Decodifica bytes de JPG/PNG para uma matriz BGR (convenção do OpenCV).

    Retorna (imagem_bgr, avisos). A orientação EXIF é aplicada porque fotos de
    celular costumam vir "deitadas" e girar só no visualizador.
    """
    if not data:
        raise ImageError("O arquivo enviado está vazio.")

    notes = []
    try:
        with Image.open(io.BytesIO(data)) as pil:
            if pil.format not in ALLOWED_FORMATS:
                raise ImageError("Formato não suportado. Envie uma imagem JPG, JPEG ou PNG.")
            pil = ImageOps.exif_transpose(pil)
            # PNG com transparência: compõe sobre fundo branco em vez de deixar o alfa virar preto
            rgba = pil.convert("RGBA")
            fundo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = np.asarray(Image.alpha_composite(fundo, rgba).convert("RGB"))
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError(
            "Não foi possível ler o arquivo como imagem. Ele pode estar corrompido "
            "ou não ser uma foto JPG/PNG válida."
        ) from exc

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    if min(h, w) < min_side:
        raise ImageError(
            f"A imagem é muito pequena ({w}×{h} px). Use uma foto com pelo menos "
            f"{min_side} px no menor lado."
        )

    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        notes.append(
            f"A imagem original ({w}×{h} px) era muito grande e foi reduzida para "
            f"{image.shape[1]}×{image.shape[0]} px antes do processamento."
        )
    return image, notes


def resize_for_detection(image: np.ndarray, max_side: int = DETECTION_SIDE):
    """Cópia reduzida usada só para achar a capa. Retorna (cópia, escala aplicada)."""
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale == 1.0:
        return image.copy(), 1.0
    # INTER_AREA evita aliasing ao reduzir
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return small, scale


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Y = 0,299 R + 0,587 G + 0,114 B (ponderação da percepção de brilho)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def gaussian_smooth(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Filtro passa-baixa gaussiano: atenua ruído de alta frequência antes do Canny.

    Sem essa etapa, o ruído e a textura fina produzem milhares de bordas falsas.
    """
    ksize = max(3, int(ksize))
    if ksize % 2 == 0:
        ksize += 1  # o núcleo precisa ter dimensão ímpar para ter centro
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)
