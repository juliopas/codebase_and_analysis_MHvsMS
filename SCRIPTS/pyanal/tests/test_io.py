"""Tests for pyanal.io.

Only `read_E_file` is exercised; the h5 functions need a pressomancy
simulation file and are skipped.
"""
import pytest

from pyanal.io import read_E_file


def test_read_E_file_finds_matching_row(tmp_path):
    path = tmp_path / "E.txt"
    path.write_text(
        "100 0.5 1 1.0 2.0 3.0 4.0\n"
        "100 1.0 1 5.0 6.0 7.0 8.0\n"
        "200 0.5 1 9.0 10.0 11.0 12.0\n"
    )
    ts, H, seed, E = read_E_file(str(path), ts=100, H=0.5, seed=1)
    assert ts == 100
    assert H == 0.5
    assert seed == 1
    assert E == (1.0, 2.0, 3.0, 4.0)


def test_read_E_file_missing_row_raises(tmp_path):
    path = tmp_path / "E.txt"
    path.write_text("100 0.5 1 1.0 2.0 3.0 4.0\n")
    with pytest.raises(ValueError):
        read_E_file(str(path), ts=99, H=0.5, seed=1)


def test_read_E_file_malformed_row_raises(tmp_path):
    path = tmp_path / "E.txt"
    path.write_text("100 0.5 1 1.0 2.0 3.0\n")  # only 6 columns
    with pytest.raises(AssertionError):
        read_E_file(str(path), ts=100, H=0.5, seed=1)


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_get_elastomer_h5():
    pass


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_get_bonds_to_parts_h5():
    pass


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_get_n_bond_to_parts():
    pass


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_elastomer_h5_ts_exists():
    pass
