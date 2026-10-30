"""Testes de detecção e do pipeline completo usando cenas SINTÉTICAS (examples/generate_examples.py)."""
import cv2
import numpy as np
import pytest

from examples.generate_examples import SCENES, build_empty_scene, build_scene
from src import cover_detection
from src.perspective import order_points
from src.pipeline import PipelineConfig, run_pipeline


@pytest.mark.parametrize("name", list(SCENES))
def test_cover_is_detected_close_to_ground_truth(name):
    image, truth = build_scene(name)
    result = run_pipeline(image)
    assert result.success and result.mode == "auto"
    error = np.abs(order_points(result.quad) - order_points(truth)).max()
    assert error < 12  # pixels, em fotos de 1200×900


def test_pipeline_output_and_steps():
    image, _ = build_scene("sintetico_01_mesa_clara")
    original = image.copy()
    result = run_pipeline(image, PipelineConfig(output_size=(400, 600)))
    assert result.final.shape == (600, 400, 3)
    keys = [s.key for s in result.steps]
    assert keys == ["original", "gray", "blur", "canny", "contours", "detected",
                    "warp", "enhance", "histogram", "resize"]
    np.testing.assert_array_equal(image, original)  # a original não é alterada
    assert "INTER" in result.interpolation_note.upper()


def test_rotated_book_is_made_portrait():
    image, _ = build_scene("sintetico_04_rotacionado")
    final = run_pipeline(image).final
    assert final.shape[0] > final.shape[1]


def test_no_book_fails_gracefully():
    result = run_pipeline(build_empty_scene())
    assert not result.success and result.mode == "failed"
    assert "automaticamente" in result.messages[0]
    assert [s.key for s in result.steps][-1] == "contours"  # etapas até a detecção continuam disponíveis


def test_manual_quad_replaces_failed_detection():
    image = build_empty_scene()
    quad = np.float32([[300, 100], [800, 120], [820, 800], [280, 780]])
    result = run_pipeline(image, manual_quad=quad)
    assert result.success and result.mode == "manual"
    assert result.final.shape == (600, 400, 3)


def test_frame_like_quad_is_rejected():
    corners = np.float32([[1, 1], [698, 2], [697, 498], [2, 497]])
    assert cover_detection.touches_frame(corners, (500, 700))
    assert not cover_detection.touches_frame(corners * 0.5 + 50, (500, 700))


def test_edge_support_follows_the_outline():
    edges = np.zeros((300, 300), np.uint8)
    cv2.rectangle(edges, (60, 40), (240, 260), 255, 1)
    on_edges = np.float32([[60, 40], [240, 40], [240, 260], [60, 260]])
    assert cover_detection.edge_support(on_edges, edges) > 0.95
    assert cover_detection.edge_support(on_edges + 40, edges) < 0.5


def test_round_object_is_not_accepted_as_a_cover():
    # disco escuro sobre fundo claro: o contorno existe, mas 4 vértices não o descrevem
    image = np.full((600, 800, 3), 235, np.uint8)
    cv2.circle(image, (400, 300), 220, (60, 80, 120), -1)
    assert not run_pipeline(image).detection.found


def test_blank_and_tiny_arrays_do_not_crash():
    for img in (np.zeros((300, 300, 3), np.uint8), np.full((300, 300, 3), 255, np.uint8)):
        assert not run_pipeline(img).success
