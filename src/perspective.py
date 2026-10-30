"""Etapas 6 e 7: ordenação dos cantos e correção de perspectiva (homografia)."""
import cv2
import numpy as np

ORIENTATIONS = ("auto", "none", "cw", "ccw", "180")


def order_points(points) -> np.ndarray:
    """Ordena 4 pontos como [superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo].

    Ordena os pontos pelo ângulo em torno do centroide (no sistema da imagem, com y para
    baixo, ângulo crescente = sentido horário) e escolhe como "superior esquerdo" o
    ponto mais próximo da origem (menor x + y). Diferente do truque soma/diferença,
    continua correto quando a capa está bastante girada.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError("São necessários exatamente 4 pontos.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def quad_dimensions(quad: np.ndarray):
    """Largura e altura do retângulo de destino: o maior de cada par de lados opostos,
    para não perder resolução do lado que está mais próximo da câmera."""
    tl, tr, br, bl = order_points(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return int(round(width)), int(round(height))


def four_point_transform(image: np.ndarray, quad, interpolation: int = cv2.INTER_CUBIC):
    """Mapeia o quadrilátero da foto para um retângulo. Retorna (imagem_retificada, matriz 3x3).

    A homografia H tem 8 graus de liberdade e é determinada exatamente por 4 pares de
    pontos. warpPerspective aplica H^-1 a cada pixel de destino e interpola a cor.
    """
    src = order_points(quad)
    width, height = quad_dimensions(src)
    if width < 10 or height < 10:
        raise ValueError("A região da capa é pequena demais para ser retificada.")
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_REPLICATE
    )
    return warped, matrix


def fix_orientation(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Gira a capa em múltiplos de 90°. No modo "auto", uma capa deitada (mais larga
    que alta) é girada para ficar em pé, já que capas de livro são verticais."""
    h, w = image.shape[:2]
    if mode == "auto":
        mode = "cw" if w > h * 1.15 else "none"
    if mode == "cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image
