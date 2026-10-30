"""BookLog Scan: interface Streamlit do pipeline de digitalização de capas."""
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src import cover_detection, enhancement, resize
from src.perspective import ORIENTATIONS, order_points
from src.pipeline import PipelineConfig, PipelineError, run_pipeline
from src.preprocessing import ImageError, load_image

try:  # seleção visual dos cantos; sem o componente, os controles deslizantes continuam funcionando
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:  # pragma: no cover
    streamlit_image_coordinates = None

logger = logging.getLogger("booklog_scan")

MAX_UPLOAD_MB = 30
PICK_WIDTH = 720  # largura máxima (px) da foto na seleção de cantos
SIZE_PRESETS = {"400 × 600 (padrão)": (400, 600), "600 × 900": (600, 900), "800 × 1200": (800, 1200)}
ORIENTATION_LABELS = {
    "auto": "Automática (deixa em pé)",
    "none": "Manter como está",
    "cw": "Girar 90° horário",
    "ccw": "Girar 90° anti-horário",
    "180": "Girar 180°",
}
FIT_LABELS = {"stretch": "Preencher todo o quadro", "fit": "Preservar proporção (com bordas)"}
CORNER_NAMES = ("Superior esquerdo", "Superior direito", "Inferior direito", "Inferior esquerdo")

# imagens nunca passam de 78% da altura da janela e mantêm a proporção
STYLE = """
<style>
[data-testid="stImage"] img { max-height: 78vh; width: auto; max-width: 100%; object-fit: contain; }
</style>
"""

st.set_page_config(
    page_title="BookLog Scan", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- utilidades
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


def stepper(current: int) -> None:
    """Indicador de progresso: Fotografia → Detecção → Resultado."""
    labels = ("Fotografia", "Detecção", "Resultado")
    st.caption("  →  ".join(f"**{i}. {name}**" if i == current else f"{i}. {name}"
                            for i, name in enumerate(labels, start=1)))


def sidebar_config() -> PipelineConfig:
    st.sidebar.header("Configurações avançadas")
    st.sidebar.caption("Opcional: os valores padrão já geram o resultado.")

    st.sidebar.subheader("Saída")
    size = SIZE_PRESETS[st.sidebar.selectbox("Tamanho final da capa", list(SIZE_PRESETS))]
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

    st.sidebar.subheader("Detecção")
    blur = st.sidebar.slider("Núcleo do filtro gaussiano", 3, 15, 5, 2)
    low = high = None
    if st.sidebar.checkbox("Definir limiares do Canny manualmente"):
        low = st.sidebar.slider("Limiar baixo", 1, 200, 20)
        high = st.sidebar.slider("Limiar alto", 2, 255, 50)

    return PipelineConfig(
        output_size=size, interpolation=interpolation, fit_mode=fit_mode, blur_ksize=blur,
        canny_low=low, canny_high=high, enhancement=method, enhancement_strength=strength,
        clip_limit=clip_limit, gamma=gamma, orientation=orientation)


# ---------------------------------------------------------------- processamento (com cache na sessão)
def cached_pipeline(image, config, file_key, manual_quad=None):
    """Evita reprocessar a cada interação (cada clique de canto reexecuta o script)."""
    suffix = None if manual_quad is None else np.round(manual_quad, 1).tobytes()
    key = (file_key, repr(config), suffix)
    slot = "auto_cache" if manual_quad is None else "manual_cache"
    cached = st.session_state.get(slot)
    if cached and cached[0] == key:
        return cached[1]
    result = run_pipeline(image, config, manual_quad=manual_quad)
    st.session_state[slot] = (key, result)
    return result


# ---------------------------------------------------------------- seleção dos cantos
def preview_with_points(image: np.ndarray, points, reference=None):
    """Foto reduzida (RGB) com os cantos já marcados, numerados 1–4. Retorna (imagem, escala)."""
    h, w = image.shape[:2]
    scale = min(1.0, PICK_WIDTH / w)
    canvas = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else image.copy()
    if reference is not None and not points:  # contorno de referência (detecção atual), discreto
        cv2.polylines(canvas, [(order_points(reference) * scale).astype(np.int32)], True, (90, 200, 90), 1, cv2.LINE_AA)
    pts = np.array(points, dtype=np.float32).reshape(-1, 2) * scale
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts.astype(np.int32)], len(pts) == 4, (0, 160, 255), 2, cv2.LINE_AA)
    radius = max(15, canvas.shape[1] // 38)
    for number, (x, y) in enumerate(pts, start=1):
        center = (int(x), int(y))
        cv2.circle(canvas, center, radius, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
        text = str(number)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, radius / 22, 2)
        cv2.putText(canvas, text, (center[0] - tw // 2, center[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    radius / 22, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale


def corner_picker(image: np.ndarray, file_key: str, reference, can_return_to_auto: bool) -> None:
    """Seleção visual: o usuário clica nos 4 cantos da capa, nesta ordem, sobre a própria foto."""
    pts_key, last_key, ver_key = f"{file_key}pts", f"{file_key}lastclick", f"{file_key}ver"
    applied_key = f"{file_key}applied"
    points = st.session_state.setdefault(pts_key, [])

    photo, panel = st.columns([3, 2])
    with photo:
        preview, _ = preview_with_points(image, points, reference)
        click = streamlit_image_coordinates(preview, key=f"{file_key}click", width="stretch")
        if click is not None:
            click_id = click.get("unix_time", (click["x"], click["y"]))
            if click_id != st.session_state.get(last_key):  # o componente repete o último clique a cada execução
                st.session_state[last_key] = click_id
                if len(points) < 4:
                    k = image.shape[1] / click["width"]  # coordenadas exibidas -> pixels da imagem original
                    points.append((click["x"] * k, click["y"] * k))
                    st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
                    st.rerun()

    with panel:
        st.markdown("**Selecione os quatro cantos da capa nesta ordem:**")
        lines = []
        for i, name in enumerate(CORNER_NAMES):
            mark = "✓" if i < len(points) else ("●" if i == len(points) else "○")
            lines.append(f"{mark} {i + 1}. {name}")
        st.markdown("  \n".join(lines))
        st.caption("Clique diretamente na fotografia." if len(points) < 4 else "Seleção completa.")
        if st.button("Aplicar correção", type="primary", disabled=len(points) < 4,
                     key=f"{file_key}apply", width="stretch"):
            st.session_state[applied_key] = np.array(points, dtype=np.float32)
            st.rerun()
        if st.button("Recomeçar seleção", key=f"{file_key}restart", width="stretch"):
            st.session_state[pts_key] = []
            st.session_state[ver_key] = st.session_state.get(ver_key, 0) + 1
            st.rerun()
        if can_return_to_auto and st.button("Voltar à detecção automática", key=f"{file_key}auto", width="stretch"):
            st.session_state.pop(applied_key, None)
            st.session_state[pts_key] = []
            st.rerun()


def slider_panel(image: np.ndarray, file_key: str, reference) -> None:
    """Ajuste preciso (fallback técnico): coordenadas dos 4 cantos em % da largura/altura da foto."""
    h, w = image.shape[:2]
    points = st.session_state.get(f"{file_key}pts", [])
    version = st.session_state.get(f"{file_key}ver", 0)  # muda quando a seleção por clique muda
    if len(points) == 4:
        default = np.array(points) / [w, h] * 100
    elif reference is not None:
        default = np.asarray(reference) / [w, h] * 100
    else:
        default = np.array([[15, 15], [85, 15], [85, 85], [15, 85]], dtype=float)
    default = np.clip(default, 0, 100)

    coords = []
    for i, name in enumerate(CORNER_NAMES):
        x_col, y_col = st.columns(2)
        x = x_col.slider(f"{i + 1}. {name} — X (%)", 0.0, 100.0, float(round(default[i][0], 1)), 0.5,
                         key=f"{file_key}x{i}v{version}")
        y = y_col.slider(f"{i + 1}. {name} — Y (%)", 0.0, 100.0, float(round(default[i][1], 1)), 0.5,
                         key=f"{file_key}y{i}v{version}")
        coords.append((x / 100 * (w - 1), y / 100 * (h - 1)))
    quad = np.array(coords, dtype=np.float32)
    show(cover_detection.draw_quad(image, quad, color=(0, 160, 255)), "Região selecionada (pré-visualização)")
    if st.button("Aplicar coordenadas", key=f"{file_key}applycoords"):
        st.session_state[f"{file_key}applied"] = quad
        st.rerun()


def selection_panel(image: np.ndarray, file_key: str, reference, can_return_to_auto: bool = False) -> None:
    if streamlit_image_coordinates is None:
        st.caption("Ajuste os quatro cantos pelas coordenadas abaixo.")
        slider_panel(image, file_key, reference)
        return
    corner_picker(image, file_key, reference, can_return_to_auto)
    with st.expander("Ajuste preciso das coordenadas"):
        slider_panel(image, file_key, reference)


# ---------------------------------------------------------------- exibição
def render_result(result, manual: bool) -> None:
    left, right = st.columns(2)
    with left:
        show(cover_detection.draw_quad(result.original, result.quad), "Fotografia original — capa selecionada")
    with right:
        show(result.final, f"Capa final — {result.final.shape[1]}×{result.final.shape[0]} px")

    status = "cantos definidos manualmente" if manual else result.messages[0]
    st.markdown(
        f"**Detecção:** {status}  \n"
        f"**Tamanho final:** {result.final.shape[1]}×{result.final.shape[0]} px  \n"
        f"**Tratamento:** {result.enhancement_note}  \n"
        f"**Interpolação:** {result.interpolation_note}"
    )
    for message in result.messages[1:]:
        if message not in (status, "Usando os cantos ajustados manualmente."):
            st.info(message)

    with st.container(border=True):
        st.markdown("**Baixar a capa processada**")
        png, jpeg = st.columns(2)
        png.download_button("Baixar PNG", encode(result.final, ".png"), "capa_booklog.png", "image/png",
                            type="primary", icon=":material/download:", width="stretch")
        jpeg.download_button("Baixar JPEG", encode(result.final, ".jpg"), "capa_booklog.jpg", "image/jpeg",
                             type="primary", icon=":material/download:", width="stretch")


def render_steps(result) -> None:
    row = []

    def flush():
        if row:
            for col, (number, step) in zip(st.columns(3), row):
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
            for col, label, stats in zip(st.columns(2), ("Antes", "Depois"), (result.stats_before, result.stats_after)):
                col.caption(f"{label}: " + " · ".join(f"{k} {v:.1f}" for k, v in stats.items()))
        else:
            row.append((number, step))
            if len(row) == 3:
                flush()
    flush()
    if result.homography is not None:
        st.caption("Matriz de homografia 3×3 utilizada na correção de perspectiva:")
        st.code(np.array2string(result.homography, precision=4, suppress_small=True))


def render_techniques() -> None:
    with st.expander("Sobre as técnicas utilizadas"):
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


# ---------------------------------------------------------------- página
def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("📚 BookLog Scan")
    st.markdown("**Transforme uma foto do seu livro em uma capa pronta para catalogação.**")
    config = sidebar_config()

    upload = st.file_uploader("Fotografia do livro (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"])
    examples = list_examples()
    example = None
    if upload is None and examples:
        choice = st.selectbox("…ou escolha uma imagem de exemplo", ["Nenhuma"] + list(examples))
        example = examples.get(choice)
    if upload is None and example is None:
        stepper(1)
        st.info("Envie uma foto" + (" ou escolha um exemplo" if examples else "") + " para começar.")
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

    if st.button("Processar capa", type="primary"):
        st.session_state["processed"] = file_key
    if st.session_state.get("processed") != file_key:
        stepper(1)
        show(image, f"Fotografia selecionada — {image.shape[1]}×{image.shape[0]} px")
        render_techniques()
        return

    applied_key = f"{file_key}applied"
    try:
        with st.spinner("Processando..."):
            auto = cached_pipeline(image, config, file_key)
            manual_quad = st.session_state.get(applied_key)
            if manual_quad is not None:
                result = cached_pipeline(image, config, file_key, manual_quad)
    except PipelineError as exc:
        st.error(str(exc))
        return
    except Exception:  # nunca expor traceback ao usuário
        logger.exception("Falha inesperada no pipeline")
        st.error("Ocorreu um erro inesperado ao processar a imagem. Tente outra foto.")
        return

    # --- detecção automática sem confiança e nada aplicado: leva direto à seleção dos cantos
    if not auto.success and manual_quad is None:
        stepper(2)
        st.info("Não foi possível identificar a capa automaticamente com confiança.")
        st.markdown("**Selecione os quatro cantos da capa para continuar.**")
        st.caption("Isso costuma ocorrer com pouco contraste, bordas fragmentadas, reflexos ou capa parcialmente visível.")
        with st.container(border=True):
            selection_panel(image, file_key, reference=None)
        with st.expander("Ver processamento técnico"):
            render_steps(auto)
        render_techniques()
        return

    # --- resultado
    if manual_quad is None:
        result = auto
    stepper(3)
    st.success("✓ Seleção manual aplicada" if manual_quad is not None else "✓ Capa identificada automaticamente")
    render_result(result, manual=manual_quad is not None)

    if st.toggle("Ajustar seleção", key=f"{file_key}adjust"):
        with st.container(border=True):
            selection_panel(image, file_key, reference=result.quad,
                            can_return_to_auto=auto.success and manual_quad is not None)

    with st.expander("Ver processamento técnico"):
        render_steps(result)
    render_techniques()


try:
    main()
except PipelineError as exc:
    st.error(str(exc))
