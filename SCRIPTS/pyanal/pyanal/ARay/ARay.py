# pyanal.ARay.ARay

import h5py
import numpy as np
import os

from .io import get_aray_file_data, write_to_file, list_available_files
from .structure import create_structure_from_flags
from .compatibility import _match_aray_to_file_aray, _a_fits_in_b, _fit_a_in_b


def _dtype_to_str_list(dtype):
    """Return `_dtype_str` as a list of (name, type_str) tuples.

    For unstructured input dtype, returns a single-field list `[('value', type_str)]`.
    For structured input, returns one tuple per field.
    """
    dtype = np.dtype(dtype)
    if dtype.fields is None:
        return [("value", str(dtype))]
    return [(name, str(typ[0])) for name, typ in dtype.fields.items()]


def _set_mask_dtype(dtype_str_list):
    """Structured bool dtype with the same field names as `dtype_str_list`."""
    return np.dtype([(name, np.bool) for name, _ in dtype_str_list])


def _apply_index(arr, index_tuple):
    """Apply a per-axis index tuple with orthogonal (outer-product) semantics.

    Scalars collapse their axis; slices and list/ndarray selectors keep it. With
    at most one advanced (list/ndarray) axis a single numpy indexing op is used,
    which is a *view* when no advanced axis is present (ints + slices = basic
    indexing) and a single copy otherwise. With two or more advanced axes the
    selection is applied axis-by-axis (np.take), so the axes combine orthogonally
    instead of broadcasting together as numpy would by default.
    """
    advanced = [i for i, sel in enumerate(index_tuple)
                if isinstance(sel, (list, np.ndarray))]
    if len(advanced) <= 1:
        return arr[index_tuple]

    result = arr
    axis = 0
    for sel in index_tuple:
        if isinstance(sel, (list, np.ndarray)):
            result = np.take(result, sel, axis=axis)
            axis += 1
        elif isinstance(sel, slice):
            result = result[(slice(None),) * axis + (sel,)]
            axis += 1
        else:  # scalar int -> collapse this axis
            result = result[(slice(None),) * axis + (sel,)]
    return result


def _deep_copy_struct(arr):
    """Return a fully independent deep copy of `arr`.

    `arr.copy()` duplicates the struct buffer, but object-dtype fields hold
    *pointers*, so a plain copy would still share the per-cell payload arrays.
    Here every object payload is itself duplicated, so mutating the result can
    never affect the original.
    """
    out = arr.copy()
    if out.dtype.fields is None:
        if out.dtype == object:
            for nidx in np.ndindex(out.shape):
                out[nidx] = np.array(out[nidx], copy=True)
    else:
        obj_fields = [name for name, (typ, _) in out.dtype.fields.items()
                      if typ == object]
        for name in obj_fields:
            field = out[name]
            for nidx in np.ndindex(field.shape):
                field[nidx] = np.array(field[nidx], copy=True)
    return out


class ARay():
    bool_type = 'bool'
    int_type = 'int16'
    float_type = 'float32'
    vlen_float = h5py.vlen_dtype(np.dtype(float_type))

    def __init__(self, shape=None, dtype=None):
        """Create a new ARay filled with sentinels and an all-False set_mask.

        For an unstructured `dtype`, `_array` stays flat (numpy semantics preserved) but `_dtype_str` is normalized to `[('value', type_str)]`.
        """
        self._vlen_names = []

        self._unset_bool = False
        self._unset_int = np.iinfo(np.dtype(self.int_type)).min # minimum possible negative integer
        self._unset_float = np.nan # not a number
        self._unset_object = np.array([], dtype=self.float_type) # object types are assumed ndarrays. Leave this one be

        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._is_unstructured = self._dtype.fields is None
        self._dtype_str = _dtype_to_str_list(self._dtype)

        self._array = np.empty(shape=shape, dtype=self._dtype)
        self._fill_sentinels(self._array, self._dtype_str, self._is_unstructured)
        for name, typ in self._dtype_str:
            if typ == "object":
                self._vlen_names.append(name)

        self._set_mask = np.zeros(shape=shape, dtype=_set_mask_dtype(self._dtype_str))

        self._mapping_dict = None
        self._file_path = None

    def _fill_sentinels(self, arr, dtype_str_list, is_unstructured):
        """Fill `arr` with the appropriate sentinel per field."""
        for name, typ in dtype_str_list:
            view = arr if is_unstructured else arr[name]
            if typ == "bool":
                view[...] = self._unset_bool
            elif typ == self.int_type:
                view[...] = self._unset_int
            elif typ == self.float_type:
                view[...] = self._unset_float
            elif typ == "object":
                for nidx in np.ndindex(self._shape):
                    if is_unstructured:
                        arr[nidx] = self._unset_object
                    else:
                        arr[nidx][name] = self._unset_object
            else:
                raise TypeError(f"Unrecognized type: {typ}")

    @classmethod
    def wrap(cls, np_array, all_set=True):
        """Wrap an existing numpy array in an ARay without re-allocating.

        `all_set=True` (default) marks every cell × field as set. Use this when
        the wrapped array is already filled with real data.
        `all_set=False` infers `set_mask` from sentinel values (NaN for float,
        int-min for int, False for bool, empty array for object).
        """
        aray_obj = cls.__new__(cls)
        aray_obj._unset_bool = False
        aray_obj._unset_int = np.iinfo(np.dtype(cls.int_type)).min
        aray_obj._unset_float = np.nan
        aray_obj._unset_object = np.array([], dtype=cls.float_type)

        aray_obj._array = np_array
        aray_obj._shape = np_array.shape
        aray_obj._dtype = np_array.dtype
        aray_obj._is_unstructured = np_array.dtype.fields is None
        aray_obj._dtype_str = _dtype_to_str_list(np_array.dtype)
        aray_obj._vlen_names = [name for name, typ in aray_obj._dtype_str if typ == "object"]

        mask_dtype = _set_mask_dtype(aray_obj._dtype_str)
        if all_set:
            aray_obj._set_mask = np.ones(np_array.shape, dtype=mask_dtype)
        else:
            aray_obj._set_mask = aray_obj._infer_set_mask_from_sentinels()

        aray_obj._mapping_dict = None
        aray_obj._file_path = None
        return aray_obj

    @classmethod
    def from_file(cls, file_path):
        (array, set_mask, mapping_dict,
         unset_bool, unset_int, unset_float, unset_object,
         vlen_names) = get_aray_file_data(file_path)

        aray_obj = cls.wrap(array, all_set=True)  # placeholder; overwrite mask below
        aray_obj._unset_bool = unset_bool
        aray_obj._unset_int = unset_int
        aray_obj._unset_float = unset_float
        aray_obj._unset_object = unset_object
        aray_obj._vlen_names = list(vlen_names) if vlen_names is not None else []

        if set_mask is None:
            # Legacy file: reconstruct from sentinels
            aray_obj._set_mask = aray_obj._infer_set_mask_from_sentinels()
        else:
            aray_obj._set_mask = set_mask

        aray_obj._mapping_dict = mapping_dict
        aray_obj._file_path = file_path
        return aray_obj

    @classmethod
    def _wrap_like(cls, parent, array, mask, mapping):
        """Build a derived ARay from an already-sliced `array`/`mask`/`mapping`.

        Pure object-construction boilerplate: it copies `parent`'s sentinels and
        re-derives dtype/flags from `array`. It does NOT decide *how* to index --
        each caller (`get`, `__getitem__`, `get_set_entries`) computes its own
        `array`, `mask` and `mapping` and passes them in.
        """
        new_aray = cls.__new__(cls)
        new_aray._unset_bool = parent._unset_bool
        new_aray._unset_int = parent._unset_int
        new_aray._unset_float = parent._unset_float
        new_aray._unset_object = parent._unset_object

        new_aray._array = array
        new_aray._shape = array.shape
        new_aray._dtype = array.dtype
        new_aray._is_unstructured = array.dtype.fields is None
        new_aray._dtype_str = _dtype_to_str_list(new_aray._dtype)
        new_aray._vlen_names = [n for n, t in new_aray._dtype_str if t == "object"]
        new_aray._set_mask = mask

        new_aray._mapping_dict = mapping
        new_aray._file_path = None
        return new_aray

    def _infer_set_mask_from_sentinels(self):
        """Build a structured bool mask: True wherever `_array` differs from the sentinel."""
        mask = np.zeros(self._array.shape, dtype=_set_mask_dtype(self._dtype_str))
        for name, typ in self._dtype_str:
            view = self._array if self._is_unstructured else self._array[name]
            if typ == "bool":
                # Sentinel is False; cannot distinguish unset False from set False. Default to unset
                mask[name] = view != self._unset_bool
            elif typ == self.int_type:
                mask[name] = view != self._unset_int
            elif typ == self.float_type:
                if np.isnan(self._unset_float):
                    mask[name] = ~np.isnan(view)
                else:
                    mask[name] = view != self._unset_float
            elif typ == "object":
                for nidx in np.ndindex(view.shape):
                    val = view[nidx]
                    mask[name][nidx] = isinstance(val, np.ndarray) and not np.array_equal(val, self._unset_object)
            else:
                raise TypeError(f"Unrecognized type: {typ}")
        return mask

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def set_mask(self):
        return self._set_mask

    @property
    def dtype_str(self):
        return tuple(self._dtype_str)

    @property
    def mapping_dict(self):
        return None if self._mapping_dict is None else dict(self._mapping_dict)

    @property
    def file_path(self):
        return self._file_path

    def coords(self, axis=None):
        """Return the coordinate labels along `axis`, in array-position order.

        The labels align one-to-one with that axis of `array`/`set_mask`. When
        only one axis is mapped, `axis` may be omitted. Requires an index mapping.
        """
        if self._mapping_dict is None:
            raise ValueError(
                "ARay has no index mapping; call add_index_mapping({...}) "
                "before requesting coords"
            )
        if axis is None:
            if len(self._mapping_dict) != 1:
                raise ValueError(
                    f"coords() needs an explicit axis name; mapped axes: {list(self._mapping_dict)}"
                )
            (axis,) = self._mapping_dict
        if axis not in self._mapping_dict:
            raise KeyError(f"Unknown axis '{axis}'; valid axes: {list(self._mapping_dict)}")
        axis_map = self._mapping_dict[axis]   # {label: position}
        return [label for label, _ in sorted(axis_map.items(), key=lambda kv: kv[1])]

    def add_index_mapping(self, mapping_dict):
        if len(mapping_dict) != self.ndim:
            raise ValueError(
                f"add_index_mapping requires one entry per axis: got {len(mapping_dict)} "
                f"entries for ndim={self.ndim}"
            )
        self._mapping_dict = {}
        for axis, (key, value_list) in enumerate(mapping_dict.items()):
            value_list = list(value_list)
            assert self._shape is not None
            assert len(value_list) == self._shape[axis], (
                f"axis {axis} ('{key}') has length {self._shape[axis]} but mapping has {len(value_list)}"
            )
            self._mapping_dict[key] = {value: i for i, value in enumerate(value_list)}

    def _select(self, coords, policy):
        """Shared selection logic for `get`/`view`/`copy`.

        `coords` maps axis name -> value(s); a scalar collapses that axis, a
        list/tuple/ndarray keeps it (multiple list axes select orthogonally). With
        no coords the whole array is taken and no index mapping is required.
        `policy` is one of "auto" (numpy decides view-vs-copy), "view" (guarantee a
        view, raising if the selection would copy) or "copy" (deep, independent copy).
        """
        if coords:
            if self._mapping_dict is None:
                raise ValueError(
                    "ARay has no index mapping; call add_index_mapping({...}) "
                    "before selecting by coordinate"
                )
            idx = self._build_indices(**coords)
        else:
            idx = (slice(None),) * self.ndim

        if policy == "view" and any(isinstance(sel, (list, np.ndarray)) for sel in idx):
            raise ValueError(
                "view() cannot return a view for a list-valued selection; "
                "use get() or copy()"
            )

        array = _apply_index(self._array, idx)
        mask = _apply_index(self._set_mask, idx)
        mapping = self._slice_mapping_dict(idx)
        if policy == "copy":
            array = _deep_copy_struct(array)
            mask = mask.copy()
        return ARay._wrap_like(self, array, mask, mapping)

    def get(self, **coords):
        """Select a sub-ARay by coordinate label(s); numpy decides view-vs-copy.

        Each coord value may be a scalar (collapses that axis) or a
        list/tuple/ndarray of values (keeps the axis; multiple list axes select
        orthogonally). With no coords the whole array is returned. The result is a
        view for a basic selection and a copy when any axis is list-valued; use
        `view()`/`copy()` to force the memory policy.
        """
        return self._select(coords, "auto")

    def view(self, **coords):
        """Like `get`, but guarantee a true view; raise if the selection would copy
        (i.e. any list-valued axis)."""
        return self._select(coords, "view")

    def copy(self, **coords):
        """Like `get`, but return a fully independent deep copy (object/vlen payloads
        duplicated too)."""
        return self._select(coords, "copy")

    def set(self, names_values_dict, **coords):
        idx = self._build_indices(**coords)
        # A single list-valued axis assigns orthogonally; two or more would need
        # np.ix_-style write semantics (not supported here).
        if sum(isinstance(sel, (list, np.ndarray)) for sel in idx) >= 2:
            raise NotImplementedError(
                "set() accepts a list of values for at most one axis at a time"
            )
        for name, value in names_values_dict.items():
            if name in self._vlen_names:
                # vlen path: iterate explicit cells
                if self._is_unstructured:
                    # Single-field unstructured object: assign via flat index
                    self._assign_vlen_unstructured(idx, value)
                else:
                    self._assign_vlen_structured(idx, name, value)
            else:
                if self._is_unstructured:
                    self._array[idx] = value
                else:
                    self._array[name][idx] = value
            # Mark set
            self._set_mask[name][idx] = True

    def _assign_vlen_unstructured(self, idx, value):
        # Iterate the positions in `idx` and write each cell
        mask = np.zeros(self._array.shape, dtype=bool)
        mask[idx] = True
        for nidx in zip(*np.where(mask)):
            self._array[nidx] = np.asarray(value, dtype=self.float_type)

    def _assign_vlen_structured(self, idx, name, value):
        mask = np.zeros(self._array.shape, dtype=bool)
        mask[idx] = True
        for nidx in zip(*np.where(mask)):
            self._array[nidx][name] = np.asarray(value, dtype=self.float_type)

    def is_set(self):
        for name, _ in self._dtype_str:
            if not self._set_mask[name].all():
                return False
        return True

    def _build_indices(self, **kwargs):
        """Convert label kwargs to a numpy index tuple (used by get/set).

        A scalar value selects (and collapses) that axis; a list/tuple/ndarray of
        values selects a subset along the axis (keeping it), in the order given, so
        `get(H=[10, 0, 3])` reorders. Axes with no kwarg are taken whole.
        """
        if self._mapping_dict is None:
            raise ValueError(
                "ARay has no index mapping; call add_index_mapping({...}) "
                "before selecting by coordinate"
            )
        unknown = [k for k in kwargs if k not in self._mapping_dict]
        if unknown:
            raise KeyError(
                f"Unknown coordinate axis/axes {unknown}; valid axes: {list(self._mapping_dict)}"
            )
        indices = []
        for axis_key in self._mapping_dict.keys():
            if axis_key in kwargs:
                value = kwargs[axis_key]
                axis_map = self._mapping_dict[axis_key]
                if isinstance(value, (list, tuple, np.ndarray)):
                    missing = [v for v in value if v not in axis_map]
                    if missing:
                        raise KeyError(f"Values {missing} not found for key '{axis_key}'")
                    indices.append([axis_map[v] for v in value])
                else:
                    if value not in axis_map:
                        raise KeyError(f"Value '{value}' not found for key '{axis_key}'")
                    indices.append(axis_map[value])
            else:
                indices.append(slice(None))  # select all along this axis
        return tuple(indices)

    def _slice_mapping_dict(self, name_or_mask_or_tuple_idx, keep_len_1_mappings=False):
        old_map = self._mapping_dict
        if old_map is None:
            return None
        new_map = {}

        if isinstance(name_or_mask_or_tuple_idx, (str, list)):
            return old_map  # getting a name, so nothing has changed
        elif isinstance(name_or_mask_or_tuple_idx, tuple):
            index_tuple = name_or_mask_or_tuple_idx
        elif isinstance(name_or_mask_or_tuple_idx, np.ndarray):
            index_tuple = np.where(name_or_mask_or_tuple_idx)
        elif isinstance(name_or_mask_or_tuple_idx, slice):
            index_tuple = (name_or_mask_or_tuple_idx,)
        elif isinstance(name_or_mask_or_tuple_idx, (int, np.integer)):
            index_tuple = (int(name_or_mask_or_tuple_idx),)
        elif name_or_mask_or_tuple_idx is None:
            return old_map
        else:
            raise NotImplementedError(
                f"Unsupported slicer type: {type(name_or_mask_or_tuple_idx)} ({name_or_mask_or_tuple_idx!r})"
            )

        # Pad with full slices if fewer axes are provided
        while len(index_tuple) < self.ndim:
            index_tuple = index_tuple + (slice(None),)

        for axis, (axis_name, axis_mapping) in enumerate(old_map.items()):
            selector = index_tuple[axis]

            # If the selector covers the whole axis, just copy the mapping
            full_slice = isinstance(selector, slice) and selector == slice(None)
            if full_slice:
                new_map[axis_name] = dict(axis_mapping)
                continue

            # Convert the selector into a list of *kept original positions*
            if isinstance(selector, slice):
                kept = list(range(*selector.indices(self.shape[axis])))
            elif isinstance(selector, (list, np.ndarray)) and np.asarray(selector).dtype == bool:
                kept = np.where(selector)[0]
            elif isinstance(selector, (list, np.ndarray)):
                kept = selector
            elif isinstance(selector, int):
                kept = [selector]
            else:
                raise TypeError(f"Unsupported selector type: {type(selector)}")

            # A scalar int collapses the axis (drop its mapping); slices and
            # list/ndarray selectors always keep the axis, even when length 1.
            keeps_axis = isinstance(selector, (slice, list, np.ndarray))
            if len(kept) < 2 and not keeps_axis and not keep_len_1_mappings:
                continue

            inverse = {idx: lbl for lbl, idx in axis_mapping.items()}
            new_map_for_axis = {}
            for new_index, old_index in enumerate(kept):
                label = inverse[old_index]
                new_map_for_axis[label] = new_index
            new_map[axis_name] = new_map_for_axis

        return new_map if new_map else None

    def write_file(self, file_path=None):
        if file_path is None:
            file_path = self._file_path
        assert file_path is not None, "File path must not be 'None"

        other_args = (
            self._array, self._set_mask, self._shape, self._dtype_str,
            self._mapping_dict,
            self._unset_bool, self._unset_int, self._unset_float, self._unset_object,
            self._vlen_names,
        )
        write_to_file(file_path, other_args)

        self._file_path = file_path
        return file_path

    def get_set_entries(self):
        if self.ndim < 1:
            raise TypeError("This function accepts only arrays with 1 or more dimensions")
        # Vectorized AND across all fields
        mask_all = np.ones(self.shape, dtype=bool)
        for name, _ in self._dtype_str:
            mask_all &= self._set_mask[name]
        # Boolean-mask selection always copies (numpy); flattens to 1-D.
        array = self._array[mask_all]
        mask = self._set_mask[mask_all]
        mapping = self._slice_mapping_dict(mask_all, keep_len_1_mappings=True)
        return ARay._wrap_like(self, array, mask, mapping)

    def find_unset_values(self):
        return [name for name, _ in self._dtype_str if not self._set_mask[name].all()]

    @classmethod
    def create_data_entry_dtype(cls, **kwargs):
        return create_structure_from_flags(cls.bool_type, cls.int_type, cls.float_type, cls.vlen_float, flag_dict=kwargs)

    def _equals(self, other):
        """True if `other` is an ARay with identical dtype, values and set_mask."""
        if self._dtype_str != other._dtype_str:
            return False
        for name, typ in self._dtype_str:
            v1 = self._array if self._is_unstructured else self._array[name]
            v2 = other._array if other._is_unstructured else other._array[name]
            if typ == "object":
                if not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(v1.flat, v2.flat)):
                    return False
            else:
                if not np.array_equal(v1, v2, equal_nan=True):
                    return False
            # set_mask must also match
            if not np.array_equal(self._set_mask[name], other._set_mask[name]):
                return False
        return True

    def __getitem__(self, key):
        """Project field(s) by name -- always a view.

        ``aray["f"]`` selects a single field (unstructured result);
        ``aray[["f1", "f2"]]`` selects several. Coordinate/positional selection
        is done with ``get(...)``, not here.
        """
        if isinstance(key, str):
            array = self._array[key] if not self._is_unstructured else self._array
            # Re-view the (flat bool) field mask as a single-field structured mask
            # so it stays uniform, sharing memory with the parent.
            mask = self._set_mask[key].view([("value", np.bool)])
        elif isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
            names = list(key)
            array = self._array[names]
            mask = self._set_mask[names]
        else:
            raise TypeError(
                "ARay[...] selects fields by name (str or list of str); "
                "use .get(axis=...) for coordinate selection"
            )
        return ARay._wrap_like(self, array, mask, self._mapping_dict)

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return len(self._array)

    def __array__(self, dtype=None):
        return self._array

    def __str__(self):
        return self._array.__str__()

    def __eq__(self, other):
        if isinstance(other, ARay):
            return self._equals(other)
        raise TypeError(
            "ARay equality is only defined against another ARay; compare aray.array "
            "for element-wise comparison, or use 'is' for identity"
        )

    def fits_in(self, other_aray):
        return _a_fits_in_b(self, other_aray)

    def place_into(self, other_aray):
        _fit_a_in_b(self, other_aray)

    def match_aray_from_file(self, filename: str, directory="", try_file_id: int | None =None, user_input: bool =True, _count: int =0, _filename_0: str | None =None, avail_files: list =[]):
        if _filename_0 is None:
            _filename_0 = filename.split(".")[0]
            avail_files = list_available_files(directory, prefix=_filename_0)

        if len(avail_files) > 0 and try_file_id is not None:
            filename = avail_files[try_file_id]

        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            return self, filename

        file_aray = ARay.from_file(path)

        return _match_aray_to_file_aray(self, file_aray, filename, directory, _filename_0=_filename_0, _count=_count, files=avail_files, try_file_id=try_file_id, user_input=user_input)
