"""Gera imagens SINTÉTICAS para testes e demonstração (não são fotografias reais).

Cada cena desenha uma capa fictícia e a posiciona com perspectiva sobre um fundo
texturizado, com ruído e iluminação irregular. O arquivo "sintetico_05_sem_livro"
não contém livro e serve para demonstrar a mensagem de falha da detecção automática.

Uso: python examples/generate_examples.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent


def make_cover(width=600, height=900, base=(120, 60, 30), accent=(230, 220, 200), seed=0):
    """Capa fictícia (BGR): degradê, moldura interna, texto e um emblema."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0.8, 1.15, height, dtype=np.float32)[:, None, None]
    cover = np.clip(np.array(base, dtype=np.float32)[None, None, :] * gradient, 0, 255)
    cover = np.repeat(cover, width, axis=1).astype(np.uint8)

    cv2.rectangle(cover, (30, 30), (width - 31, height - 31), accent, 3)  # moldura interna
    cv2.putText(cover, "TITULO", (70, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, accent, 5, cv2.LINE_AA)
    cv2.putText(cover, "DE EXEMPLO", (70, 280), cv2.FONT_HERSHEY_DUPLEX, 1.8, accent, 3, cv2.LINE_AA)
    cv2.circle(cover, (width // 2, 540), 130, accent, -1)
    cv2.circle(cover, (width // 2, 540), 90, base, -1)
    for i in range(4):  # linhas que imitam o texto da quarta capa
        y = 760 + i * 28
        cv2.line(cover, (70, y), (int(width - 70 - rng.integers(0, 160)), y), accent, 6)
    return cover


def _background(size, color, rng, stripes=True):
    w, h = size
    low = rng.normal(0, 1, (6, 8)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)[:, :, None]
    bg = np.array(color, dtype=np.float32)[None, None, :] * (1 + 0.08 * low)
    if stripes:  # veios de madeira
        x = np.arange(w, dtype=np.float32)[None, :, None]
        bg = bg * (1 + 0.05 * np.sin(x / 9.0 + 3 * low))
    return np.clip(bg, 0, 255).astype(np.float32)


def make_scene(cover, quad, size=(1200, 900), bg_color=(150, 190, 215), seed=0,
               noise=6.0, light=(0.75, 1.1), shadow=True, stripes=True):
    """Posiciona `cover` no quadrilátero `quad` (TL, TR, BR, BL) de uma cena.

    Retorna (imagem_bgr, quad_verdadeiro).
    """
    rng = np.random.default_rng(seed)
    w, h = size
    ch, cw = cover.shape[:2]
    src = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    dst = np.float32(quad)
    matrix = cv2.getPerspectiveTransform(src, dst)

    scene = _background(size, bg_color, rng, stripes)
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), matrix, size).astype(np.float32) / 255
    warped = cv2.warpPerspective(cover, matrix, size, flags=cv2.INTER_LINEAR).astype(np.float32)

    if shadow:
        offset = np.float32([[1, 0, 14], [0, 1, 18]])
        shade = cv2.GaussianBlur(cv2.warpAffine(mask, offset, size), (0, 0), 14)
        scene *= (1 - 0.45 * shade)[:, :, None]
    scene = scene * (1 - mask[:, :, None]) + warped * mask[:, :, None]

    # iluminação irregular: gradiente diagonal
    gx = np.linspace(light[0], light[1], w, dtype=np.float32)[None, :]
    gy = np.linspace(1.0, 0.92, h, dtype=np.float32)[:, None]
    scene *= (gx * gy)[:, :, None]

    scene = cv2.GaussianBlur(scene, (3, 3), 0) + rng.normal(0, noise, scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8), dst


SCENES = {
    # nome: (quad TL,TR,BR,BL, parâmetros da capa, parâmetros da cena)
    "sintetico_01_mesa_clara": (
        [(330, 110), (800, 150), (830, 800), (300, 760)],
        dict(base=(120, 60, 30), accent=(230, 220, 200), seed=1),
        dict(bg_color=(150, 190, 215), seed=1),
    ),
    "sintetico_02_perspectiva_forte": (
        [(260, 150), (880, 60), (940, 840), (200, 720)],
        dict(base=(40, 40, 170), accent=(240, 240, 240), seed=2),
        dict(bg_color=(55, 60, 60), seed=2, light=(0.6, 1.15)),
    ),
    "sintetico_03_baixo_contraste": (
        [(350, 120), (790, 130), (810, 790), (330, 780)],
        dict(base=(175, 205, 215), accent=(110, 140, 160), seed=3),
        dict(bg_color=(190, 200, 205), seed=3, shadow=False, stripes=False, noise=4.0),
    ),
    "sintetico_04_rotacionado": (
        [(420, 70), (900, 260), (740, 800), (250, 640)],
        dict(base=(60, 130, 60), accent=(225, 235, 225), seed=4),
        dict(bg_color=(120, 150, 175), seed=4),
    ),
}


def build_scene(name):
    quad, cover_kw, scene_kw = SCENES[name]
    return make_scene(make_cover(**cover_kw), quad, **scene_kw)


def build_empty_scene(seed=5):
    """Cena sem livro: só fundo texturizado e ruído."""
    rng = np.random.default_rng(seed)
    scene = _background((1200, 900), (140, 170, 190), rng)
    scene = scene + rng.normal(0, 6.0, scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8)


def main():
    for name in SCENES:
        image, _ = build_scene(name)
        cv2.imwrite(str(OUT_DIR / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(OUT_DIR / "sintetico_05_sem_livro.jpg"), build_empty_scene(),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Imagens sintéticas gravadas em {OUT_DIR}")


if __name__ == "__main__":
    main()
