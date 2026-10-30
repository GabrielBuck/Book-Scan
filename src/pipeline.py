"""Orquestra o pipeline completo e registra cada etapa intermediária para exibição.

Entrada -> grayscale -> filtragem -> bordas -> contornos -> perspectiva
        -> intensidade/histograma -> interpolação -> resultado
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import cover_detection, edge_detection, enhancement, histogram, perspective, preprocessing, resize


class PipelineError(RuntimeError):
    """Falha durante o processamento, com mensagem apresentável ao usuário."""


@dataclass
class PipelineConfig:
    output_size: tuple = resize.DEFAULT_SIZE
    interpolation: str = "auto"
    fit_mode: str = "stretch"
    blur_ksize: int = 5
    canny_low: int | None = None    # None = limiares padrão (edge_detection.DEFAULT_*)
    canny_high: int | None = None
    enhancement: str = "clahe"
    enhancement_strength: float = 1.0
    clip_limit: float = 2.0
    gamma: float = 1.0
    orientation: str = "auto"


@dataclass
class Step:
    key: str
    title: str
    image: np.ndarray       # BGR ou escala de cinza
    description: str
    wide: bool = False      # ocupa a largura toda na interface (ex.: histogramas)


@dataclass
class PipelineResult:
    original: np.ndarray
    steps: list = field(default_factory=list)
    detection: cover_detection.DetectionResult | None = None
    quad: np.ndarray | None = None          # cantos usados, em pixels da imagem original
    mode: str = "failed"                    # "auto", "manual" ou "failed"
    final: np.ndarray | None = None
    warped: np.ndarray | None = None
    homography: np.ndarray | None = None
    enhancement_note: str = ""
    interpolation_note: str = ""
    stats_before: dict = field(default_factory=dict)
    stats_after: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.final is not None


def run_pipeline(image: np.ndarray, config: PipelineConfig | None = None, manual_quad=None) -> PipelineResult:
    """Executa todas as etapas. `manual_quad` (4x2, pixels da imagem original) substitui a detecção automática."""
    config = config or PipelineConfig()
    result = PipelineResult(original=image)
    steps = result.steps

    # 1. Original (preservada intacta; as etapas de detecção trabalham em cópias)
    steps.append(Step("original", "Original", image,
                      "Fotografia enviada, usada na correção de perspectiva e no resultado final."))
    small, scale = preprocessing.resize_for_detection(image)

    # 2. Escala de cinza
    gray = preprocessing.to_grayscale(small)
    steps.append(Step("gray", "Escala de cinza", gray,
                      "Y = 0,299R + 0,587G + 0,114B. Um único canal basta para achar bordas e "
                      "reduz o custo do processamento."))

    # 3. Filtragem espacial
    blurred = preprocessing.gaussian_smooth(gray, config.blur_ksize)
    steps.append(Step("blur", "Suavização gaussiana", blurred,
                      f"Filtro passa-baixa gaussiano {config.blur_ksize}×{config.blur_ksize}: "
                      "atenua ruído e textura fina que gerariam bordas falsas."))

    # 4. Bordas
    edges = edge_detection.detect_edges(blurred, small, config.blur_ksize, config.canny_low, config.canny_high)
    steps.append(Step("canny", "Bordas (Canny)", edges.edges,
                      f"Gradiente + supressão de não-máximos + histerese com limiares "
                      f"{edges.low}/{edges.high}, aplicados ao cinza e a cada canal de cor."))

    # 5. Contornos e escolha da capa
    detection = cover_detection.find_cover(edges.edges_closed, blurred, edges.edges)
    result.detection = detection
    steps.append(Step("contours", "Contornos candidatos",
                      cover_detection.draw_contours(small, detection),
                      "Azul: maiores contornos encontrados. Amarelo: os que viraram quadriláteros "
                      "convexos plausíveis (tamanho, ângulos e proporção)."))

    # 6. Escolha do quadrilátero e ordenação dos cantos
    if manual_quad is not None:
        quad = perspective.order_points(manual_quad)
        result.mode = "manual"
        result.messages.append("Usando os cantos ajustados manualmente.")
    elif detection.found:
        quad = detection.quad / scale  # volta para as coordenadas da imagem completa
        result.mode = "auto"
        result.messages.append(detection.message)
    else:
        result.messages.append(detection.message)
        return result

    result.quad = quad
    steps.append(Step("detected", "Capa detectada",
                      cover_detection.draw_quad(image, quad),
                      "Cantos ordenados em TL, TR, BR e BL — a ordem exigida para montar a "
                      "correspondência com o retângulo de destino."))

    _process_cover(image, quad, config, result)
    return result


def _process_cover(image: np.ndarray, quad: np.ndarray, config: PipelineConfig, result: PipelineResult) -> None:
    steps = result.steps
    try:
        # 7. Perspectiva (sobre a imagem colorida original, em resolução total)
        warped, matrix = perspective.four_point_transform(image, quad)
        warped = perspective.fix_orientation(warped, config.orientation)
        result.warped, result.homography = warped, matrix
        steps.append(Step("warp", "Perspectiva corrigida", warped,
                          f"Homografia 3×3 (cv2.getPerspectiveTransform) aplicada com "
                          f"cv2.warpPerspective: {warped.shape[1]}×{warped.shape[0]} px."))

        # 8. Tratamento de intensidade (luminância)
        enhanced, result.enhancement_note = enhancement.enhance(
            warped, config.enhancement, config.enhancement_strength,
            config.clip_limit, gamma=config.gamma)
        steps.append(Step("enhance", "Tratamento de intensidade", enhanced, result.enhancement_note + "."))

        # 9. Histogramas antes/depois
        result.stats_before = histogram.histogram_stats(warped)
        result.stats_after = histogram.histogram_stats(enhanced)
        steps.append(Step("histogram", "Histogramas de luminância",
                          histogram.render_histograms(warped, enhanced, config.enhancement),
                          "Mesma escala vertical nos dois gráficos. A linha tracejada é a "
                          "distribuição acumulada.", wide=True))

        # 10. Interpolação e redimensionamento
        final, key, result.interpolation_note = resize.resize_cover(
            enhanced, config.output_size, config.interpolation, config.fit_mode)
        result.final = final
        steps.append(Step("resize", "Resultado redimensionado", final,
                          f"{final.shape[1]}×{final.shape[0]} px. Interpolação: {result.interpolation_note}"))
        if warped.shape[1] < config.output_size[0] or warped.shape[0] < config.output_size[1]:
            result.messages.append(
                "A capa na foto tem resolução menor que o tamanho de saída; o resultado foi ampliado "
                "e pode ficar menos nítido. Aproxime a câmera para obter melhor qualidade.")
    except (cv2.error, ValueError) as exc:
        raise PipelineError(
            "Não foi possível corrigir a perspectiva da região selecionada. "
            "Verifique se os quatro cantos formam um quadrilátero válido."
        ) from exc
