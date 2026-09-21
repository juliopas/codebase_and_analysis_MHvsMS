"""Target-behavior tests for pyanal.ARay

Run with: pytest pyanal/tests/test_aray.py -v
"""
import warnings

import numpy as np
import pytest

from pyanal import ARay
from pyanal.ARay.compatibility import _a_fits_in_b, _fit_a_in_b


# ---------- constructor ----------

def test_constructor_structured_sets_nothing():
    dtype = [("a", "int16"), ("b", "float32")]
    a = ARay((3, 2), dtype=dtype)
    assert a.shape == (3, 2)
    assert a._dtype_str == [("a", "int16"), ("b", "float32")]
    # nothing set yet
    assert not a.is_set()
    assert a.find_unset_values() == ["a", "b"]
    # sentinels filled
    assert np.all(a._array["a"] == np.iinfo(np.int16).min)
    assert np.all(np.isnan(a._array["b"]))


def test_constructor_unstructured_keeps_flat_array_normalizes_dtype_str():
    a = ARay((4,), dtype=np.float32)
    # _array stays flat — must still behave like a plain numpy array
    assert a._array.shape == (4,)
    assert a._array.dtype == np.float32
    np.testing.assert_array_equal(a._array + 0, a._array)  # arithmetic works
    # _dtype_str normalized
    assert a._dtype_str == [("value", "float32")]


def test_constructor_with_bool_field():
    a = ARay((3,), dtype=[("flag", "bool")])
    assert not a.is_set()
    # Sentinel for bool is False; nothing has been *set* yet
    assert a._array["flag"].dtype == np.bool


def test_constructor_with_object_field():
    a = ARay((3,), dtype=[("v", "object")])
    assert not a.is_set()
    # Every cell initialized to the unset object (empty float32 array)
    for nidx in np.ndindex(a.shape):
        assert isinstance(a._array[nidx]["v"], np.ndarray)


# ---------- set/get ----------

def test_set_updates_value_and_set_mask():
    a = ARay((3,), dtype=[("x", "float32"), ("y", "int16")])
    a.add_index_mapping({"i": [10, 20, 30]})
    a.set({"x": 1.5}, i=20)
    assert a._array["x"][1] == np.float32(1.5)
    # y is still unset, but x at i=20 is set
    assert a.get(i=20)["x"].is_set()
    assert not a.get(i=20)["y"].is_set()


def test_set_bool_to_false_is_still_set():
    """The ambiguous bool case: setting False must register as set."""
    a = ARay((2,), dtype=[("flag", "bool")])
    a.add_index_mapping({"i": [0, 1]})
    a.set({"flag": False}, i=0)
    # flag at i=0 was set, even though the value is False
    assert a.get(i=0).is_set()
    assert not a.get(i=1).is_set()


def test_set_object_to_empty_array_is_still_set():
    """Setting an empty array must register as set."""
    a = ARay((2,), dtype=[("v", "object")])
    a.add_index_mapping({"i": [0, 1]})
    a.set({"v": np.array([], dtype=np.float32)}, i=0)
    # Even though value matches the sentinel, it was explicitly set
    assert a.get(i=0).is_set()
    assert not a.get(i=1).is_set()


# ---------- mapping ----------

def test_add_index_mapping_requires_all_axes():
    a = ARay((3, 2), dtype=[("x", "float32")])
    # Missing one axis must raise
    with pytest.raises(ValueError):
        a.add_index_mapping({"i": [0, 1, 2]})


def test_get_with_mapping():
    a = ARay((3, 2), dtype=[("x", "float32")])
    a.add_index_mapping({"i": [10, 20, 30], "j": ["A", "B"]})
    a.set({"x": 7.5}, i=20, j="B")
    sub = a.get(i=20, j="B")
    # sub is a 0-d slice; access the scalar
    assert float(sub._array["x"]) == np.float32(7.5)


# ---------- coords ----------

def test_coords_single_axis_in_position_order():
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    # sole axis -> no name needed; labels in array-position order
    assert a.coords() == [0, 3, 6, 10]
    assert a.coords("H") == [0, 3, 6, 10]


def test_coords_reflects_subset_order():
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    assert a.get(H=[10, 0, 3]).coords() == [10, 0, 3]


def test_coords_multi_axis_requires_name():
    a = ARay((3, 2), dtype=[("x", "float32")])
    a.add_index_mapping({"i": [10, 20, 30], "j": ["A", "B"]})
    with pytest.raises(ValueError):
        a.coords()
    assert a.coords("i") == [10, 20, 30]
    assert a.coords("j") == ["A", "B"]


def test_coords_without_mapping_raises():
    a = ARay.wrap(np.zeros(3, dtype=[("x", "float32")]))
    with pytest.raises(ValueError):
        a.coords()


def test_coords_unknown_axis_raises():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    with pytest.raises(KeyError):
        a.coords("nope")


# ---------- list-valued get / subsetting ----------

def test_get_list_keeps_axis_in_requested_order():
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    for i, H in enumerate([0, 3, 6, 10]):
        a.set({"x": float(H)}, H=H)
    sub = a.get(H=[10, 0, 3])
    # axis kept (not collapsed), length matches the requested subset
    assert sub.shape == (3,)
    np.testing.assert_array_equal(sub.array["x"], np.float32([10.0, 0.0, 3.0]))
    # mapping rebuilt in requested order
    assert sub.mapping_dict["H"] == {10: 0, 0: 1, 3: 2}


def test_get_single_element_list_keeps_length_1_axis():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    a.set({"x": 9.0}, H=3)
    sub = a.get(H=[3])
    assert sub.shape == (1,)
    assert sub.mapping_dict["H"] == {3: 0}
    assert sub.get(H=3).is_set()


def test_get_scalar_collapses_axis_to_0d():
    # A scalar selection collapses its axis, even when it is the only axis: the
    # result is a 0-d scalar cell (intended behavior).
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    a.set({"x": 5.0}, H=6)
    sub = a.get(H=6)
    assert sub.shape == ()
    assert float(sub.array["x"]) == np.float32(5.0)


def test_get_scalar_collapses_axis_and_keeps_view():
    # Collapse one axis with a scalar while keeping another (the realistic case:
    # get(ts, DENS, K, HEIGHT) collapses those, keeps H). Basic indexing -> view,
    # so subsetting never duplicates the underlying buffer.
    a = ARay((2, 3), dtype=[("x", "float32")])
    a.add_index_mapping({"ts": [100, 200], "H": [0, 3, 6]})
    a.set({"x": 5.0}, ts=200, H=3)
    sub = a.get(ts=200)
    # scalar collapses ts; H axis kept
    assert sub.shape == (3,)
    # basic indexing -> a true view onto the parent buffer (no copy)
    assert np.shares_memory(a.array["x"], sub.array["x"])


def test_get_list_shares_object_payload_by_reference():
    a = ARay((3,), dtype=[("v", "object")])
    a.add_index_mapping({"H": [0, 3, 6]})
    payload = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a.set({"v": payload}, H=3)
    sub = a.get(H=[0, 3])
    # the struct buffer is copied (advanced indexing) but the object payload
    # is the *same* ndarray, so heavy data is shared, not duplicated
    assert sub.array["v"][1] is a.array["v"][1]


def test_get_two_list_axes_orthogonal_subset():
    a = ARay((3, 3), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 1, 2], "ts": [10, 20, 30]})
    for i, H in enumerate([0, 1, 2]):
        for j, ts in enumerate([10, 20, 30]):
            a.set({"x": float(10 * i + j)}, H=H, ts=ts)
    sub = a.get(H=[0, 2], ts=[10, 30])
    # orthogonal (outer-product) subset, not numpy's broadcast-together default
    assert sub.shape == (2, 2)
    np.testing.assert_array_equal(
        sub.array["x"], np.float32([[0.0, 2.0], [20.0, 22.0]])
    )


def test_get_list_with_missing_value_raises():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    with pytest.raises(KeyError):
        a.get(H=[0, 99])


def test_subset_then_get_set_entries_aligns():
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    # set only some of the requested subset
    a.set({"x": 0.0}, H=0)
    a.set({"x": 6.0}, H=6)
    subset = [0, 3, 6]
    sub = a.get(H=subset)
    H_list_plot = [H for H in subset if sub.get(H=H).is_set()]
    vals = sub.get_set_entries().array
    assert len(H_list_plot) == len(vals)
    assert H_list_plot == [0, 6]
    np.testing.assert_array_equal(vals["x"], np.float32([0.0, 6.0]))


def test_set_list_kwarg_writes_and_marks_set():
    a = ARay((4,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6, 10]})
    a.set({"x": 7.0}, H=[0, 6])
    assert a.get(H=0).is_set()
    assert a.get(H=6).is_set()
    assert not a.get(H=3).is_set()
    np.testing.assert_array_equal(
        a.array["x"][[0, 2]], np.float32([7.0, 7.0])
    )


def test_set_two_list_axes_raises():
    a = ARay((3, 3), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 1, 2], "ts": [10, 20, 30]})
    with pytest.raises(NotImplementedError):
        a.set({"x": 1.0}, H=[0, 1], ts=[10, 20])


# ---------- view / copy policy ----------

def test_view_shares_memory_for_basic_selection():
    a = ARay((2, 3), dtype=[("x", "float32")])
    a.add_index_mapping({"ts": [100, 200], "H": [0, 3, 6]})
    a.set({"x": 5.0}, ts=200, H=3)
    sub = a.view(ts=200)
    assert sub.shape == (3,)
    assert np.shares_memory(a.array["x"], sub.array["x"])


def test_view_raises_for_list_selection():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    with pytest.raises(ValueError):
        a.view(H=[0, 3])


def test_view_and_copy_no_coords_act_on_whole_array():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    a.set({"x": 5.0}, H=3)
    # whole-array view shares memory
    whole_view = a.view()
    assert whole_view.shape == (3,)
    assert np.shares_memory(a.array["x"], whole_view.array["x"])
    # whole-array copy is an independent ARay (not a bare numpy array)
    whole_copy = a.copy()
    assert isinstance(whole_copy, ARay)
    assert not np.shares_memory(a.array["x"], whole_copy.array["x"])


def test_copy_is_fully_independent():
    a = ARay((2,), dtype=[("x", "float32"), ("v", "object")])
    a.add_index_mapping({"H": [0, 3]})
    a.set({"x": 1.0, "v": np.array([1.0, 2.0, 3.0], dtype=np.float32)}, H=0)
    sub = a.copy()
    # scalar field is detached
    assert not np.shares_memory(a.array["x"], sub.array["x"])
    sub.array["x"][0] = 999.0
    assert a.array["x"][0] == np.float32(1.0)
    # object/vlen payload is deep-copied too
    assert sub.array["v"][0] is not a.array["v"][0]
    sub.array["v"][0][0] = -1.0
    assert a.array["v"][0][0] == np.float32(1.0)


def test_get_no_coords_works_without_mapping_but_coord_selection_raises():
    arr = np.zeros((3,), dtype=[("x", "float32")])
    a = ARay.wrap(arr)  # no mapping_dict
    # no coords -> whole-array view, no mapping needed
    whole = a.get()
    assert whole.shape == (3,)
    assert np.shares_memory(a.array["x"], whole.array["x"])
    # coordinate selection without a mapping -> informative error
    with pytest.raises(ValueError):
        a.get(i=0)


# ---------- __getitem__ (field selection only) ----------

def test_getitem_field_selection_is_a_view():
    a = ARay((3,), dtype=[("x", "float32"), ("y", "int16")])
    a.add_index_mapping({"H": [0, 3, 6]})
    a.set({"x": 1.0, "y": 2}, H=0)
    single = a["x"]
    assert np.shares_memory(a.array["x"], single.array)
    multi = a[["x", "y"]]
    assert np.shares_memory(a.array["x"], multi.array["x"])


def test_getitem_rejects_non_field_keys():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    for bad in (0, slice(0, 2), (0, 1), [0, 1]):
        with pytest.raises(TypeError):
            a[bad]


# ---------- "one way to do it": no numpy leak, no silent misuse ----------

def test_shape_ndim_dtype_properties():
    a = ARay((2, 3), dtype=[("x", "float32"), ("y", "int16")])
    assert a.shape == (2, 3)
    assert a.ndim == 2
    assert a.dtype == np.dtype([("x", "float32"), ("y", "int16")])


def test_no_numpy_attribute_delegation():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    a.set({"x": 1.0}, H=0)
    # numpy ops no longer leak through the wrapper
    for attr in ("mean", "reshape", "astype", "ravel", "T"):
        assert not hasattr(a, attr)
    # the sanctioned escape hatch is .array
    assert np.nanmax(a["x"].array) == np.float32(1.0)


def test_eq_only_against_aray():
    a = ARay((2,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3]})
    b = ARay((2,), dtype=[("x", "float32")])
    b.add_index_mapping({"H": [0, 3]})
    assert (a == b) is True
    a.set({"x": 1.0}, H=0)
    assert (a == b) is False
    # comparing to non-ARay raises instead of returning a numpy bool array
    with pytest.raises(TypeError):
        a == 5


def test_get_set_name_array_removed():
    a = ARay((3,), dtype=[("x", "float32")])
    assert not hasattr(a, "get_set_name_array")


def test_unknown_coord_kwarg_raises():
    a = ARay((3,), dtype=[("x", "float32")])
    a.add_index_mapping({"H": [0, 3, 6]})
    with pytest.raises(KeyError):
        a.get(Hh=3)          # typo'd axis name
    with pytest.raises(KeyError):
        a.set({"x": 1.0}, Hh=3)


# ---------- wrap ----------

def test_wrap_default_marks_all_set_and_avoids_double_allocation():
    arr = np.zeros((5,), dtype=[("x", "float32"), ("y", "int16")])
    a = ARay.wrap(arr)
    # Same array under the hood (no copy/realloc)
    assert a._array is arr
    # Default: all set
    assert a.is_set()


def test_wrap_all_set_false_infers_from_sentinels():
    arr = np.empty((3,), dtype=[("x", "float32")])
    arr["x"][0] = np.nan
    arr["x"][1] = 1.0
    arr["x"][2] = 2.0
    a = ARay.wrap(arr, all_set=False)
    a.add_index_mapping({"i": [0, 1, 2]})
    # nan position is unset; others are set
    assert not a.get(i=0).is_set()
    assert a.get(i=1).is_set()
    assert a.get(i=2).is_set()


# ---------- file roundtrip ----------

def test_write_load_roundtrip(tmp_path):
    a = ARay((2, 3), dtype=[("x", "float32"), ("y", "int16")])
    a.add_index_mapping({"i": [0, 1], "j": [10, 20, 30]})
    a.set({"x": 1.5, "y": 42}, i=1, j=20)
    a.set({"x": 2.5}, i=0, j=30)

    path = str(tmp_path / "aray.h5")
    a.write_file(path)

    b = ARay.from_file(path)
    assert b.shape == (2, 3)
    assert b._dtype_str == [("x", "float32"), ("y", "int16")]
    # Values preserved
    assert b._array["x"][1, 1] == np.float32(1.5)
    assert b._array["y"][1, 1] == np.int16(42)
    assert b._array["x"][0, 2] == np.float32(2.5)
    # Set mask preserved
    assert b.get(i=1, j=20).is_set()
    assert not b.get(i=0, j=10).is_set()
    assert b.get(i=0, j=30)["x"].is_set()
    assert not b.get(i=0, j=30)["y"].is_set()


def test_legacy_file_load_reconstructs_set_mask(tmp_path):
    """File written without a /set_mask dataset (legacy layout) should
    still load, with set_mask inferred from sentinels and a warning emitted."""
    import h5py

    path = str(tmp_path / "legacy.h5")
    # Write a minimal ARay h5 by hand, no /ARay/set_mask
    dtype = np.dtype([("x", "float32")])
    arr = np.empty((3,), dtype=dtype)
    arr["x"][0] = np.nan  # unset
    arr["x"][1] = 1.0     # set
    arr["x"][2] = np.nan  # unset
    with h5py.File(path, "w") as f:
        g = f.require_group("ARay")
        g.attrs["shape"] = (3,)
        g.attrs["dtype"] = [("x", "float32")]
        g.attrs["unset_bool"] = False
        g.attrs["unset_int"] = np.iinfo(np.int16).min
        g.attrs["unset_float"] = np.nan
        g.attrs["unset_object"] = np.array([], dtype=np.float32)
        g.attrs["vlen_names"] = []
        g.create_dataset("array", data=arr, dtype=dtype)
        mg = g.require_group("mapping_dict")
        mg.attrs["mapping_dict_keys"] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = ARay.from_file(path)
        assert any("set_mask" in str(w.message).lower() for w in caught), (
            f"Expected a legacy-set_mask warning, got: {[str(w.message) for w in caught]}"
        )
    # Reconstructed set_mask from sentinels (nan -> unset)
    np.testing.assert_array_equal(b.set_mask["x"], [False, True, False])


# ---------- compatibility / fitting ----------

def test_fits_in_and_fit_a_in_b():
    a = ARay((2, 1), dtype=[("x", "float32"), ("y", "int16")])
    a.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    a.set({"x": 1.5, "y": 7}, ts=100, DENS=0.5)
    a.set({"x": 2.5, "y": 9}, ts=200, DENS=0.5)

    b = ARay((3, 2), dtype=[("x", "float32"), ("y", "int16")])
    b.add_index_mapping({"ts": [0, 100, 200], "DENS": [0.5, 1.0]})

    assert _a_fits_in_b(a, b)
    assert not _a_fits_in_b(b, a)
    _fit_a_in_b(a, b)
    # values transferred
    assert b._array["x"][1, 0] == np.float32(1.5)
    assert b._array["y"][1, 0] == np.int16(7)
    assert b._array["x"][2, 0] == np.float32(2.5)
    assert b._array["y"][2, 0] == np.int16(9)
    # set_mask transferred
    assert b.get(ts=100, DENS=0.5).is_set()
    assert b.get(ts=200, DENS=0.5).is_set()
    # untouched cells still unset
    assert not b.get(ts=0, DENS=0.5).is_set()


def test_fit_a_in_b_object_field_merges_payloads():
    """Ragged object/vlen payloads must land in the right cells under the
    advanced-index assignment, with set_mask transferred."""
    a = ARay((2, 1), dtype=[("v", "object")])
    a.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    a.set({"v": np.array([1., 2., 3.], dtype=np.float32)}, ts=100, DENS=0.5)
    a.set({"v": np.array([4., 5.], dtype=np.float32)}, ts=200, DENS=0.5)

    b = ARay((3, 2), dtype=[("v", "object")])
    b.add_index_mapping({"ts": [0, 100, 200], "DENS": [0.5, 1.0]})

    _fit_a_in_b(a, b)
    assert np.array_equal(b._array["v"][1, 0], np.array([1., 2., 3.], dtype=np.float32))
    assert np.array_equal(b._array["v"][2, 0], np.array([4., 5.], dtype=np.float32))
    # set_mask transferred only to the matching cells
    assert b.get(ts=100, DENS=0.5).is_set()
    assert b.get(ts=200, DENS=0.5).is_set()
    assert not b.get(ts=0, DENS=0.5).is_set()
    assert not b.get(ts=100, DENS=1.0).is_set()


def test_match_aray_from_file_fits_file_into_superset(tmp_path):
    """Full write->load->fit round-trip: an old file whose dtype lacks a field
    is merged into a fresh in-memory ARay that adds the field (the
    rms_interpolated upgrade scenario), via the non-interactive auto-fit path."""
    # "old" file: dtype without the new field
    old = ARay((2, 1), dtype=[("x", "float32")])
    old.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    old.set({"x": 1.5}, ts=100, DENS=0.5)
    old.set({"x": 2.5}, ts=200, DENS=0.5)
    old.write_file(str(tmp_path / "old.h5"))

    # "new" in-memory ARay: same axes, adds a field
    new = ARay((2, 1), dtype=[("x", "float32"), ("rms_interpolated", "float32")])
    new.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})

    merged, filename = new.match_aray_from_file(
        filename="old.h5", directory=str(tmp_path), try_file_id=0, user_input=False)

    assert merged is new                 # in-memory superset is returned, not the file
    assert filename == "old.h5"
    # old entries copied in
    assert merged._array["x"][0, 0] == np.float32(1.5)
    assert merged.get(ts=100, DENS=0.5)["x"].is_set()
    assert merged.get(ts=200, DENS=0.5)["x"].is_set()
    # new field exists and stays unset (nothing to copy from the old file)
    assert "rms_interpolated" in [n for n, _ in merged._dtype_str]
    assert not merged.get(ts=100, DENS=0.5)["rms_interpolated"].is_set()


def test_fit_a_in_b_warns_on_overlap():
    """Overwriting destination cells that are already set must warn; a wins."""
    a = ARay((2, 1), dtype=[("x", "float32")])
    a.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    a.set({"x": 1.5}, ts=100, DENS=0.5)

    b = ARay((2, 1), dtype=[("x", "float32")])
    b.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    b.set({"x": 9.9}, ts=100, DENS=0.5)   # already set where a is also set

    with pytest.warns(UserWarning, match="overwrote"):
        _fit_a_in_b(a, b)
    assert b._array["x"][0, 0] == np.float32(1.5)   # overwrite: a wins


def test_fit_a_in_b_no_warning_when_destination_empty():
    """The normal fresh-load case (empty destination) must not warn."""
    a = ARay((2, 1), dtype=[("x", "float32")])
    a.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})
    a.set({"x": 1.5}, ts=100, DENS=0.5)

    b = ARay((2, 1), dtype=[("x", "float32")])
    b.add_index_mapping({"ts": [100, 200], "DENS": [0.5]})

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        _fit_a_in_b(a, b)


# ---------- perf sanity (cheap) ----------

def test_get_set_entries_vectorized_is_fast():
    """Sanity: get_set_entries on a 10^4-cell × 20-field ARay finishes quickly."""
    import time
    fields = [(f"f{i}", "float32") for i in range(20)]
    a = ARay((10_000,), dtype=fields)
    a.add_index_mapping({"i": list(range(10_000))})
    # set half the entries
    for name, _ in fields:
        a._array[name][:5000] = 1.0
        a._set_mask[name][:5000] = True

    t0 = time.perf_counter()
    sub = a.get_set_entries()
    dt = time.perf_counter() - t0
    assert sub.shape[0] == 5000
    assert dt < 1.0, f"get_set_entries took {dt:.3f}s; expected <1s"
