"""Smoke tests for pyanal.plot. Visual correctness is not asserted."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyanal.plot import (
    create_cm,
    create_mm,
    darken_color,
    plot_ridgeline,
    plt_close,
    plt_scatter,
    save_plot_simple,
)


def test_create_cm_samples_has_expected_keys():
    cm = create_cm(samples=True)
    for k in [
        '0.20-hard', '0.20-soft', '0.25-hard', '0.25-soft', '0.30-hard', '0.30-soft',
        'hard-4', 'hard-20', 'soft-4', 'soft-20',
    ]:
        assert k in cm


def test_create_cm_default_factory_returns_default_color():
    cm = create_cm(samples=False, default=lambda: "blue")
    assert cm['unknown_key'] == "blue"


def test_create_cm_H_list_maps_distinct():
    cm = create_cm(samples=False, H_list=[0, 1, 2])
    assert {cm[0], cm[1], cm[2]} == set([cm[0], cm[1], cm[2]])  # distinct values
    # 3 distinct H values -> 3 entries
    assert len({cm[0], cm[1], cm[2]}) == 3


def test_create_cm_H_list_too_long_raises():
    with pytest.raises(AssertionError):
        create_cm(samples=False, H_list=list(range(17)))


def test_create_mm_samples_has_expected_keys():
    mm = create_mm(samples=True)
    assert '0.20-hard' in mm
    assert 'soft-20' in mm


def test_darken_color_halves_intensity():
    r, g, b = darken_color("red", 0.5)  # "red" = (1,0,0) -> (0.5, 0, 0)
    assert r == pytest.approx(0.5)
    assert g == pytest.approx(0.0)
    assert b == pytest.approx(0.0)


def test_plt_close_all_then_keep_one():
    plt.close("all")
    f1 = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    plt_close("all", but_list=[f1.number])
    remaining = list(plt.get_fignums())
    assert remaining == [f1.number]
    plt.close("all")


def test_plt_scatter_3d_with_size_raises():
    plt.close("all")
    fig = plt.figure()
    fig.add_subplot(projection="3d")
    with pytest.raises(NotImplementedError):
        plt_scatter(plt, [0, 1], [0, 1], data_z=[0, 1], size=0.5)
    plt.close("all")


def test_save_plot_simple_writes_file(tmp_path):
    plt.close("all")
    fig = plt.figure(num="testfig")
    plt.plot([0, 1], [0, 1])
    out_base = str(tmp_path / "out")
    save_plot_simple("testfig", out_base, format=".png", xlabel="x", ylabel="y", grid=True)
    out_path = out_base + ".png"
    import os
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
    plt.close("all")


def test_plot_ridgeline_returns_fig_ax():
    plt.close("all")
    data_list = [np.random.randn(50) for _ in range(3)]
    fig, ax = plot_ridgeline(
        data_list, H_values=[0, 1, 2], colors=["red", "blue", "green"],
        bins=np.linspace(-3, 3, 11),
    )
    assert fig is not None
    assert ax is not None
    plt.close("all")
