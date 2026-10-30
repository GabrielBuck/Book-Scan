"""BookLog Scan: interface Streamlit do pipeline de digitalização de capas."""
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src import cover_detection, enhancement, resize
from src.perspective import ORIENTATIONS
from src.pipeline import PipelineConfig, PipelineError, run_pipeline
from src.preprocessing import ImageError, load_image

logger = logging.getLogger("booklog_scan")

MAX_UPLOAD_MB = 30
SIZE_PRESETS = {"400 × 600 (padrão)": (400, 600), "600 × 900": (600, 900), "800 × 1200": (800, 1200)}
ORIENTATION_LABELS = {
    "auto": "Automática (deixa em pé)",
    "none": "Manter como está",
    "cw": "Girar 90° horário",
    "ccw": "Girar 90° anti-horário",
    "180": "Girar 180°",
}
FIT_LABELS = {"stretch": "Preencher todo o quadro", "fit": "Preservar proporção (com bordas)"}

st.set_page_config(page_title="BookLog Scan", page_icon="📚", layout="wide")


def list_examples() -> dict:
    """Imagens de examples/real/ (fotos do grupo) e examples/sintetico_* (sintéticas), por nome."""
    base = Path(__file__).parent / "examples"
    files = sorted((base / "real").glob("*")) + sorted(base.glob("sintetico_*"))
    return {
        (f"{f.name} (sintética)" if f.parent == base else f.name): f
        for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    }


def show(image: np.ndarray, caption: str | None = None) -> None:
    """Exibe imagem BGR (ou cinza) do OpenCV."""
    st.image(image, caption=caption, channels="BGR" if image.ndim == 3 else "RGB", width="stretch")


def encode(image: np.ndarray, fmt: str) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt == ".jpg" else []
    ok, buffer = cv2.imencode(fmt, image, params)
    if not ok:
        raise PipelineError("Não foi possível gerar o arquivo para download.")
    return buffer.tobytes()


def sidebar_config() -> PipelineConfig:
    st.sidebar.header("Configurações")
    size = SIZE_PRESETS[st.sidebar.selectbox("Tamanho da capa final", list(SIZE_PRESETS))]
    fit_mode = st.sidebar.radio("Ajuste ao quadro", list(FIT_LABELS), format_func=FIT_LABELS.get)
    interpolation = st.sidebar.selectbox(
        "Interpolação", list(resize.INTERPOLATIONS), format_func=lambda k: resize.INTERPOLATIONS[k][0])
    orientation = st.sidebar.selectbox(
        "Orientação da capa", ORIENTATIONS, format_func=ORIENTATION_LABELS.get)

    st.sidebar.subheader("Tratamento de intensidade")
    method = st.sidebar.selectbox("Método", list(enhancement.METHODS), format_func=enhancement.METHODS.get)
    strength, clip_limit, gamma = 1.0, 2.0, 1.0
    if method != "none":
        strength = st.sidebar.slider("Intensidade do efeito", 0.0, 1.0, 1.0, 0.05)
    if method == "clahe":
        clip_limit = st.sidebar.slider("Limite de contraste (clipLimit)", 1.0, 6.0, 2.0, 0.5)
    if method == "gamma":
        gamma = st.sidebar.slider("γ (menor que 1 clareia)", 0.4, 2.5, 0.8, 0.05)

    with st.sidebar.expander("Detecção (avançado)"):
        blur = st.slider("Núcleo do filtro gaussiano", 3, 15, 5, 2)
        custom = st.checkbox("Definir limiares do Canny manualmente")
        low = high = None
        if custom:
            low = st.slider("Limiar baixo", 1, 200, 20)
            high = st.slider("Limiar alto", 2, 255, 50)

    return PipelineConfig(
        output_size=size, interpolation=interpolation, fit_mode=fit_mode, blur_ksize=blur,
        canny_low=low, canny_high=high, enhancement=method, enhancement_strength=strength,
        clip_limit=clip_limit, gamma=gamma, orientation=orientation)


def manual_quad_ui(image: np.ndarray, default_quad, file_key: str):
    """Ajuste manual dos 4 cantos (percentuais da imagem). Retorna os cantos em pixels ou None."""
    h, w = image.shape[:2]
    if default_quad is None:
        default = np.array([[15, 15], [85, 15], [85, 85], [15, 85]], dtype=float)
    else:
        default = np.asarray(default_quad) / [w, h] * 100
    default = np.clip(default, 0, 100)

    cols = st.columns([3, 2])
    points = []
    with cols[0]:
        grid = st.columns(4)
        for i, name in enumerate(("Superior esquerdo", "Superior direito", "Inferior direito", "Inferior esquerdo")):
            with grid[i]:
                st.caption(name)
                x = st.slider("X (%)", 0.0, 100.0, float(round(default[i][0], 1)), 0.5, key=f"{file_key}x{i}")
                y = st.slider("Y (%)", 0.0, 100.0, float(round(default[i][1], 1)), 0.5, key=f"{file_key}y{i}")
                points.append((x / 100 * (w - 1), y / 100 * (h - 1)))
    quad = np.array(points, dtype=np.float32)
    with cols[1]:
        show(cover_detection.draw_quad(image, quad, color=(0, 160, 255)), "Região selecionada (pré-visualização)")
    return quad


def render_steps(result) -> None:
    st.subheader("3. Etapas do processamento")
    row = []

    def flush():
        if row:
            cols = st.columns(3)
            for col, (number, step) in zip(cols, row):
                with col:
                    show(step.image, f"{number}. {step.title}")
                    st.caption(step.description)
            row.clear()

    for number, step in enumerate(result.steps, start=1):
        if step.wide:
            flush()
            st.markdown(f"**{number}. {step.title}**")
            show(step.image)
            st.caption(step.description)
            cols = st.columns(2)
            for col, label, stats in ((cols[0], "Antes", result.stats_before), (cols[1], "Depois", result.stats_after)):
                col.caption(f"{label}: " + " · ".join(f"{k} {v:.1f}" for k, v in stats.items()))
        else:
            row.append((number, step))
            if len(row) == 3:
                flush()
    flush()


def render_result(result) -> None:
    st.subheader("4. Resultado")
    left, right = st.columns(2)
    with left:
        show(cover_detection.draw_quad(result.original, result.quad), "Fotografia original (capa selecionada)")
    with right:
        show(result.final, f"Capa final — {result.final.shape[1]}×{result.final.shape[0]} px")

    st.info(
        f"**Tratamento:** {result.enhancement_note}  \n"
        f"**Interpolação:** {result.interpolation_note}"
    )
    for message in result.messages[1:]:
        st.warning(message)

    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Baixar PNG", encode(result.final, ".png"), "capa_booklog.png", "image/png", type="primary")
    c2.download_button("Baixar JPEG", encode(result.final, ".jpg"), "capa_booklog.jpg", "image/jpeg")
    with st.expander("Matriz de homografia 3×3 utilizada"):
        st.code(np.array2string(result.homography, precision=4, suppress_small=True))


def render_techniques() -> None:
    st.divider()
    st.subheader("Técnicas de Computação Visual utilizadas")
    st.markdown(
        """
| Etapa | Técnica | Conteúdo da disciplina |
|---|---|---|
| Escala de cinza | Média ponderada dos canais (Y = 0,299R + 0,587G + 0,114B) | Modelos de cor |
| Suavização | Filtro gaussiano (convolução com núcleo passa-baixa) | Filtragem espacial |
| Bordas | Canny: gradiente de Sobel, supressão de não-máximos e histerese | Detecção de bordas |
| Contornos | Contornos, fecho convexo, aproximação poligonal e validação geométrica | Segmentação / descrição de regiões |
| Perspectiva | Homografia 3×3 a partir de 4 pontos (`getPerspectiveTransform` + `warpPerspective`) | Transformações geométricas |
| Tratamento | CLAHE / equalização / alongamento / gama sobre a luminância (canal L do LAB) | Transformações de intensidade |
| Histograma | Histograma e distribuição acumulada da luminância antes e depois | Histograma |
| Redimensionamento | Interpolação (`INTER_AREA` ao reduzir, `INTER_CUBIC` ao ampliar) | Interpolação |
"""
    )


def main() -> None:
    st.title("📚 BookLog Scan")
    st.markdown("**Transforme uma foto do seu livro em uma capa pronta para catalogação.**")
    st.caption(
        "Protótipo web de uma futura funcionalidade do BookLog — Computação Visual, Universidade "
        "Presbiteriana Mackenzie. Envie uma foto com a capa inteira visível e o pipeline "
        "mostra cada etapa até gerar a imagem padronizada."
    )
    config = sidebar_config()

    st.subheader("1. Fotografia do livro")
    upload = st.file_uploader("Envie uma imagem JPG, JPEG ou PNG", type=["jpg", "jpeg", "png"])
    examples = list_examples()
    example = None
    if upload is None and examples:
        choice = st.selectbox("…ou use uma imagem de exemplo", ["Nenhuma"] + list(examples))
        example = examples.get(choice)
    if upload is None and example is None:
        st.info("Aguardando uma imagem. Envie uma foto" + (" ou escolha um exemplo." if examples else "."))
        render_techniques()
        return

    data = upload.getvalue() if upload is not None else example.read_bytes()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"O arquivo tem mais de {MAX_UPLOAD_MB} MB. Envie uma imagem menor.")
        return
    try:
        image, notes = load_image(data)
    except ImageError as exc:
        st.error(str(exc))
        return

    file_key = hashlib.md5(data).hexdigest()[:10]
    for note in notes:
        st.info(note)
    show(image, f"Imagem original — {image.shape[1]}×{image.shape[0]} px")

    st.subheader("2. Processamento")
    if st.button("Processar capa", type="primary"):
        st.session_state["processed"] = file_key
    if st.session_state.get("processed") != file_key:
        render_techniques()
        return

    try:
        with st.spinner("Processando..."):
            auto = run_pipeline(image, config)
    except PipelineError as exc:
        st.error(str(exc))
        return
    except Exception:  # nunca expor traceback ao usuário
        logger.exception("Falha inesperada no pipeline")
        st.error("Ocorreu um erro inesperado ao processar a imagem. Tente outra foto.")
        return

    applied_key = f"{file_key}applied"  # cantos confirmados pelo usuário
    if auto.success:
        st.success(auto.messages[0])
        panel = st.expander("Ajustar os cantos manualmente (opcional)")
    else:
        st.info(
            "Não foi possível identificar a capa automaticamente com confiança. "
            "Ajuste os quatro cantos da capa para continuar."
        )
        st.caption("Isso costuma ocorrer com pouco contraste, bordas fragmentadas, reflexos ou capa parcialmente visível.")
        panel = st.container(border=True)

    with panel:
        st.caption(
            "Posicione os quatro cantos sobre a capa (em % da largura e da altura da foto), confira a "
            "pré-visualização e clique em **Aplicar correção**."
        )
        corners = manual_quad_ui(image, auto.quad, file_key)
        apply_col, reset_col, _ = st.columns([2, 3, 2])
        if apply_col.button("Aplicar correção", type="primary", key=f"{file_key}apply"):
            st.session_state[applied_key] = corners
        if auto.success and applied_key in st.session_state:
            if reset_col.button("Voltar à detecção automática", key=f"{file_key}reset"):
                del st.session_state[applied_key]
    manual_quad = st.session_state.get(applied_key)

    result = auto
    if manual_quad is not None:
        try:
            result = run_pipeline(image, config, manual_quad=manual_quad)
        except PipelineError as exc:
            st.error(str(exc))
            return
        except Exception:
            logger.exception("Falha no pipeline com cantos manuais")
            st.error("Não foi possível processar com esses cantos. Ajuste os pontos e tente novamente.")
            return

    if not result.success:
        st.caption("O processamento continua assim que os cantos forem aplicados. Etapas já executadas:")
        render_steps(result)
        return
    if result.mode == "manual":
        st.success(result.messages[0])
    render_steps(result)
    render_result(result)
    render_techniques()


try:
    main()
except PipelineError as exc:
    st.error(str(exc))
