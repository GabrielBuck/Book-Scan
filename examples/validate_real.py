"""Roda o pipeline em todas as fotografias de uma pasta e gera uma tabela-resumo.

Uso (a partir da raiz do repositório):
    python examples/validate_real.py                  # usa examples/real/
    python examples/validate_real.py outra_pasta      # qualquer pasta com JPG/PNG

Saídas em validation_output/: resultados.md, resultados.csv, capas/ (resultado final em PNG)
e deteccao/ (foto com a capa detectada), para conferência visual.
"""
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.preprocessing import ImageError, load_image  # noqa: E402
from src.cover_detection import draw_quad  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png"}
COLUMNS = ["arquivo", "resolução", "método", "score", "detecção automática", "fallback",
           "dimensões após perspectiva", "resultado final", "observações"]
METHOD_LABELS = {"canny": "Canny (cinza + B/G/R)", "otsu": "Otsu (plano B)"}


def evaluate_image(path: Path, config: PipelineConfig | None = None):
    """Processa uma foto e devolve (linha_da_tabela, resultado_do_pipeline | None, imagem | None)."""
    row = dict.fromkeys(COLUMNS, "—")
    row["arquivo"] = path.name
    try:
        image, notes = load_image(path.read_bytes())
    except ImageError as exc:
        row["detecção automática"] = "não"
        row["observações"] = f"Arquivo recusado: {exc}"
        return row, None, None

    h, w = image.shape[:2]
    row["resolução"] = f"{w}×{h}"
    result = run_pipeline(image, config or PipelineConfig())
    detection = result.detection
    observations = list(notes)

    if detection.found:
        row["método"] = METHOD_LABELS.get(detection.source, detection.source)
        row["score"] = f"{detection.score:.2f}"
        row["detecção automática"] = "sim"
        row["fallback"] = "Otsu" if detection.source == "otsu" else "não"
        observations.append(f"capa ocupa {detection.candidates[0].area_ratio:.0%} da foto")
    else:
        row["detecção automática"] = "não"
        row["fallback"] = "ajuste manual necessário"
        observations.append("nenhuma região plausível encontrada")

    if result.warped is not None:
        row["dimensões após perspectiva"] = f"{result.warped.shape[1]}×{result.warped.shape[0]}"
    if result.success:
        row["resultado final"] = f"{result.final.shape[1]}×{result.final.shape[0]} (OK)"
        observations += result.messages[1:]
    row["observações"] = "; ".join(observations)
    return row, result, image


def main(argv) -> int:
    folder = Path(argv[1]) if len(argv) > 1 else ROOT / "examples" / "real"
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTENSIONS) if folder.is_dir() else []
    if not files:
        print(f"Nenhuma imagem JPG/PNG encontrada em {folder}.\n"
              "Coloque as fotografias reais do grupo em examples/real/ e execute novamente.")
        return 1

    out = ROOT / "validation_output"
    (out / "capas").mkdir(parents=True, exist_ok=True)
    (out / "deteccao").mkdir(parents=True, exist_ok=True)
    for old in [*(out / "capas").glob("*"), *(out / "deteccao").glob("*")]:
        old.unlink()  # evita sobras de execuções anteriores

    rows = []
    for path in files:
        row, result, image = evaluate_image(path)
        rows.append(row)
        if result is not None and result.quad is not None:
            cv2.imwrite(str(out / "deteccao" / f"{path.stem}.jpg"), draw_quad(image, result.quad))
        if result is not None and result.success:
            cv2.imwrite(str(out / "capas" / f"{path.stem}.png"), result.final)
        print(f"{path.name}: detecção automática = {row['detecção automática']}")

    header = "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
    body = "".join("| " + " | ".join(r[c].replace("|", "/") for c in COLUMNS) + " |\n" for r in rows)
    ok = sum(r["detecção automática"] == "sim" for r in rows)
    summary = f"\n{ok} de {len(rows)} fotos tiveram a capa detectada automaticamente.\n"
    (out / "resultados.md").write_text(header + body + summary, encoding="utf-8")
    with open(out / "resultados.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + header + body + summary)
    print(f"Tabela e imagens gravadas em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
