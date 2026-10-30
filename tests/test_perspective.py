import cv2
import numpy as np
import pytest

from src.perspective import fix_orientation, four_point_transform, order_points, quad_dimensions

TL, TR, BR, BL = (100, 50), (400, 80), (420, 500), (80, 470)


def test_order_points_from_any_permutation():
    expected = np.array([TL, TR, BR, BL], dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = expected[rng.permutation(4)]
        np.testing.assert_array_equal(order_points(shuffled), expected)


def test_order_points_rotated_quad():
    # quadrado girado 30°: o truque soma/diferença falharia, o ordenamento angular não
    angle = np.radians(30)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 100
    pts = square @ rot.T + 300
    ordered = order_points(pts[[2, 0, 3, 1]])
    assert ordered[0].sum() == ordered.sum(axis=1).min()  # começa pelo mais próximo da origem
    # sentido horário: o produto vetorial entre arestas consecutivas é positivo (y para baixo)
    edges = np.roll(ordered, -1, axis=0) - ordered
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    assert (cross > 0).all()


def test_order_points_requires_four_points():
    with pytest.raises(ValueError):
        order_points([(0, 0), (1, 1), (2, 2)])


def test_quad_dimensions_uses_longest_opposite_sides():
    assert quad_dimensions(np.array([(0, 0), (100, 0), (100, 200), (0, 200)])) == (100, 200)
    w, h = quad_dimensions(np.array([TL, TR, BR, BL]))
    assert w == round(max(np.hypot(300, 30), np.hypot(340, 30)))


def test_four_point_transform_recovers_flat_pattern():
    # padrão conhecido (quatro faixas coloridas) distorcido por uma homografia e depois retificado
    flat = np.zeros((300, 200, 3), np.uint8)
    flat[:150, :100], flat[:150, 100:] = (255, 0, 0), (0, 255, 0)
    flat[150:, :100], flat[150:, 100:] = (0, 0, 255), (255, 255, 0)
    corners = np.float32([[0, 0], [199, 0], [199, 299], [0, 299]])
    dst = np.float32([[60, 40], [330, 90], [310, 420], [30, 380]])
    skewed = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(corners, dst), (400, 480))

    warped, matrix = four_point_transform(skewed, dst)
    assert matrix.shape == (3, 3)
    h, w = warped.shape[:2]
    # o centro de cada quadrante deve ter a cor original
    for (fy, fx), color in {(0.25, 0.25): (255, 0, 0), (0.25, 0.75): (0, 255, 0),
                            (0.75, 0.25): (0, 0, 255), (0.75, 0.75): (255, 255, 0)}.items():
        assert np.abs(warped[int(fy * h), int(fx * w)].astype(int) - color).max() < 40


def test_four_point_transform_rejects_degenerate_quad():
    with pytest.raises(ValueError):
        four_point_transform(np.zeros((100, 100, 3), np.uint8), [(5, 5), (6, 5), (6, 6), (5, 6)])


def test_fix_orientation():
    wide = np.zeros((100, 200, 3), np.uint8)
    assert fix_orientation(wide, "auto").shape[:2] == (200, 100)
    assert fix_orientation(np.zeros((200, 100, 3), np.uint8), "auto").shape[:2] == (200, 100)
    assert fix_orientation(wide, "none").shape[:2] == (100, 200)
    assert fix_orientation(wide, "180").shape[:2] == (100, 200)
