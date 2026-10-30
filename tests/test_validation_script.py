import tempfile
from pathlib import Path

import cv2

from examples.generate_examples import build_empty_scene, build_scene
from examples.validate_real import COLUMNS, evaluate_image


def test_evaluate_image_success_failure_and_invalid_file():
    with tempfile.TemporaryDirectory() as folder:
        _check_rows(Path(folder))


def _check_rows(tmp_path: Path):
    good = tmp_path / "boa.jpg"
    cv2.imwrite(str(good), build_scene("sintetico_01_mesa_clara")[0])
    row, result, _ = evaluate_image(good)
    assert set(row) == set(COLUMNS)
    assert row["detecção automática"] == "sim" and row["resultado final"] == "400×600 (OK)"
    assert result.success

    empty = tmp_path / "sem_livro.jpg"
    cv2.imwrite(str(empty), build_empty_scene())
    row, result, _ = evaluate_image(empty)
    assert row["detecção automática"] == "não" and "manual" in row["fallback"]

    broken = tmp_path / "quebrado.png"
    broken.write_bytes(b"nao sou imagem")
    row, result, image = evaluate_image(broken)
    assert result is None and "recusado" in row["observações"]
