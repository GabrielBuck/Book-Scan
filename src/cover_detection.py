"""Etapa 5: busca da capa entre os contornos da imagem.

Nenhum contorno é aceito só por ser o maior. Cada candidato precisa virar um
quadrilátero convexo e passar por validações geométricas (tamanho, ângulos,
proporção); os aprovados recebem uma pontuação e vence o melhor.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .perspective import order_points

MIN_AREA_RATIO = 0.10      # a capa deve ocupar ao menos 10% da foto
MAX_AREA_RATIO = 0.985     # acima disso, o "quadrilátero" é provavelmente a moldura da foto
ANGLE_RANGE = (55.0, 125.0)  # ângulos internos aceitos (perspectiva deforma os 90°)
MIN_SIDE_RATIO = 0.35      # lados opostos não podem diferir demais
ASPECT_RANGE = (0.40, 2.50)  # largura/altura da capa retificada
MIN_SOLIDITY = 0.85        # área do contorno / área do fecho convexo (rejeita formas côncavas)
MIN_FIT = 0.90             # área do quadrilátero / área do fecho: o contorno precisa ser bem descrito por 4 vértices
MIN_SUPPORT = 0.50         # fração do perímetro do quadrilátero que coincide com bordas detectadas
BORDER_MARGIN = 0.02       # quadrilátero com os 4 cantos a < 2% da borda da foto é a moldura da foto
SUPPORT_TOLERANCE = 4      # px (na imagem de detecção) entre o perímetro e uma borda para contar como coincidência
MAX_CONTOURS = 12          # só os maiores contornos são avaliados


@dataclass
class Candidate:
    quad: np.ndarray        # 4x2, ordenado TL, TR, BR, BL
    score: float
    area_ratio: float
    source: str
    support: float = 0.0


@dataclass
class DetectionResult:
    quad: np.ndarray | None = None
    score: float = 0.0
    source: str = "none"    # "canny", "otsu" ou "none"
    candidates: list = field(default_factory=list)
    contours: list = field(default_factory=list)  # contornos avaliados, para desenhar
    message: str = ""

    @property
    def found(self) -> bool:
        return self.quad is not None


def polygon_area(quad) -> float:
    return float(abs(cv2.contourArea(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2))))


def interior_angles(quad: np.ndarray) -> np.ndarray:
    """Ângulo interno (graus) em cada vértice de um quadrilátero ordenado."""
    angles = []
    for i in range(4):
        prev_v = quad[i - 1] - quad[i]
        next_v = quad[(i + 1) % 4] - quad[i]
        cos = np.dot(prev_v, next_v) / (np.linalg.norm(prev_v) * np.linalg.norm(next_v) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.array(angles)


def approximate_quad(hull: np.ndarray):
    """Simplifica o fecho convexo com Douglas-Peucker, aumentando a tolerância até
    sobrarem 4 vértices. Retorna None se não for possível chegar a um quadrilátero."""
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            return None
    return None


def edge_support(quad: np.ndarray, edges: np.ndarray) -> float:
    """Fração do perímetro do quadrilátero que está a até SUPPORT_TOLERANCE px de uma borda
    detectada. Um contorno de capa real segue bordas da imagem; o fecho de uma região de
    textura ou um contorno "inventado" pela aproximação poligonal, não."""
    perimeter = np.zeros(edges.shape, np.uint8)
    cv2.polylines(perimeter, [quad.astype(np.int32)], True, 255, 1)
    size = 2 * SUPPORT_TOLERANCE + 1
    near_edge = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    on_perimeter = perimeter > 0
    return float((on_perimeter & (near_edge > 0)).sum() / max(on_perimeter.sum(), 1))


def touches_frame(quad: np.ndarray, image_shape) -> bool:
    """True se os 4 cantos estão colados nas bordas da foto (moldura da imagem, não uma capa)."""
    h, w = image_shape[:2]
    mx, my = BORDER_MARGIN * w, BORDER_MARGIN * h
    near = (quad[:, 0] < mx) | (quad[:, 0] > w - 1 - mx) | (quad[:, 1] < my) | (quad[:, 1] > h - 1 - my)
    return bool(near.all())


def score_quad(quad: np.ndarray, solidity: float, fit: float, image_shape) -> tuple[float, float]:
    """Valida um quadrilátero ordenado. Retorna (pontuação, razão_de_área); pontuação 0 = rejeitado."""
    h, w = image_shape[:2]
    area_ratio = polygon_area(quad) / float(h * w)
    if not (MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO) or solidity < MIN_SOLIDITY:
        return 0.0, area_ratio
    if fit < MIN_FIT or touches_frame(quad, image_shape):
        return 0.0, area_ratio

    angles = interior_angles(quad)
    if angles.min() < ANGLE_RANGE[0] or angles.max() > ANGLE_RANGE[1]:
        return 0.0, area_ratio
    angle_score = 1.0 - float(np.abs(angles - 90.0).mean()) / 35.0

    tl, tr, br, bl = quad
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    if min(top, bottom) / max(top, bottom) < MIN_SIDE_RATIO:
        return 0.0, area_ratio
    if min(left, right) / max(left, right) < MIN_SIDE_RATIO:
        return 0.0, area_ratio

    aspect = ((top + bottom) / 2) / ((left + right) / 2 + 1e-9)
    if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
        return 0.0, area_ratio
    # capas típicas têm proporção ~0,6–0,8 (ou o inverso, se o livro estiver deitado)
    typical = 0.55 <= aspect <= 0.90 or 1.10 <= aspect <= 1.80
    aspect_score = 1.0 if typical else 0.7

    # preferência por regiões grandes, mas sem premiar a moldura da foto inteira
    size_score = area_ratio if area_ratio < 0.95 else area_ratio * 0.5
    return size_score * solidity * max(angle_score, 0.0) * aspect_score, area_ratio


def find_candidates(binary: np.ndarray, source: str, edges: np.ndarray):
    """Extrai contornos de uma imagem binária e devolve (candidatos válidos, contornos avaliados).

    `edges` é o mapa de bordas fino usado para medir o suporte de cada quadrilátero."""
    contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    min_area = MIN_AREA_RATIO * binary.shape[0] * binary.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    candidates = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        quad = approximate_quad(hull)
        if quad is None:
            continue
        quad = order_points(quad)
        solidity = cv2.contourArea(contour) / hull_area
        fit = polygon_area(quad) / hull_area
        score, area_ratio = score_quad(quad, solidity, fit, binary.shape)
        if score <= 0:
            continue
        support = edge_support(quad, edges)
        if support >= MIN_SUPPORT:
            candidates.append(Candidate(quad, score * support, area_ratio, source, support))
    return candidates, contours


def segmentation_masks(blurred: np.ndarray):
    """Plano B para fotos de baixo contraste: limiarização de Otsu nas duas polaridades
    (capa mais clara ou mais escura que o fundo), com abertura/fechamento morfológico."""
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for m in (mask, cv2.bitwise_not(mask)):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        yield cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)


def find_cover(edges_closed: np.ndarray, blurred: np.ndarray, edges: np.ndarray) -> DetectionResult:
    """Procura a capa nas bordas do Canny; se nada plausível surgir, tenta a segmentação por Otsu.

    `edges` é o mapa de bordas fino (antes do fechamento), usado para medir o suporte."""
    candidates, contours = find_candidates(edges_closed, "canny", edges)

    if not candidates:
        for mask in segmentation_masks(blurred):
            cands, conts = find_candidates(mask, "otsu", edges)
            candidates += cands
            contours += conts

    best_source = max(candidates, key=lambda c: c.score).source if candidates else "none"
    result = DetectionResult(contours=contours, source=best_source)
    if not candidates:
        result.message = (
            "Não foi possível identificar a capa automaticamente com confiança. "
            "Isso costuma acontecer com pouco contraste entre capa e fundo, bordas "
            "fragmentadas, cantos escondidos ou fundo muito texturizado."
        )
        return result

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    result.candidates = candidates
    result.quad, result.score = best.quad, best.score
    origem = "bordas (Canny)" if best.source == "canny" else "segmentação por limiar (Otsu)"
    result.message = (
        f"Capa detectada via {origem}; ocupa {best.area_ratio:.0%} da foto "
        f"(pontuação {best.score:.2f}, {best.support:.0%} do contorno sobre bordas)."
    )
    return result


def draw_contours(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Cópia da imagem com os contornos avaliados (azul) e os candidatos aprovados (amarelo)."""
    canvas = image.copy()
    thickness = max(2, image.shape[1] // 400)
    cv2.drawContours(canvas, result.contours, -1, (255, 140, 30), thickness)
    for cand in result.candidates:
        cv2.polylines(canvas, [cand.quad.astype(np.int32)], True, (0, 220, 255), thickness)
    return canvas


def draw_quad(image: np.ndarray, quad: np.ndarray, color=(60, 200, 60)) -> np.ndarray:
    """Cópia da imagem com o quadrilátero escolhido e os cantos rotulados (TL, TR, BR, BL)."""
    canvas = image.copy()
    quad = order_points(quad)
    thickness = max(3, image.shape[1] // 300)
    cv2.polylines(canvas, [quad.astype(np.int32)], True, color, thickness)
    scale = max(0.6, image.shape[1] / 1200)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), quad):
        cv2.circle(canvas, (int(x), int(y)), thickness * 3, (0, 0, 255), -1)
        cv2.putText(canvas, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 255), max(2, thickness - 1), cv2.LINE_AA)
    return canvas
