# Aray.io

import os
import warnings

import h5py
import numpy as np


_LEGACY_WARN_EMITTED = False


def _warn_legacy():
    global _LEGACY_WARN_EMITTED
    if not _LEGACY_WARN_EMITTED:
        warnings.warn(
            "ARay file lacks /ARay/set_mask; reconstructing from sentinels. "
            "Re-save the file (aray.write_file(path)) to upgrade the layout.",
            stacklevel=3,
        )
        # _LEGACY_WARN_EMITTED = True


def _dtype_attr_to_list(dtype_attr):
    """h5 attrs round-trip dtype_str as a numpy array of tuples (bytes, bytes).
    Normalize back to a Python list of (str, str)."""
    if isinstance(dtype_attr, np.ndarray):
        out = []
        for entry in dtype_attr:
            name = entry[0]
            typ = entry[1]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(typ, bytes):
                typ = typ.decode("utf-8")
            out.append((str(name), str(typ)))
        return out
    if isinstance(dtype_attr, (list, tuple)):
        return [(str(n), str(t)) for n, t in dtype_attr]
    raise TypeError(f"Unrecognized dtype attr type: {type(dtype_attr)}")


def get_aray_file_data(file_path):
    with h5py.File(file_path, "r") as h5_file:
        ARay_grp = h5_file["ARay"]
        shape = ARay_grp.attrs["shape"]
        dtype_attr = ARay_grp.attrs["dtype"]

        unset_bool = ARay_grp.attrs["unset_bool"]
        unset_int = ARay_grp.attrs["unset_int"]
        unset_float = ARay_grp.attrs["unset_float"]
        unset_object = ARay_grp.attrs["unset_object"]
        vlen_names = ARay_grp.attrs["vlen_names"]

        array = np.asarray(ARay_grp["array"])

        # Sanity-check the stored dtype matches the actual array's dtype fields
        dtype_str_list = _dtype_attr_to_list(dtype_attr)
        if array.dtype.fields is not None:
            arr_fields = [(name, str(typ[0])) for name, typ in array.dtype.fields.items()]
            assert arr_fields == dtype_str_list, (
                f"On-disk dtype attr {dtype_str_list} disagrees with array dtype {arr_fields}"
            )
        else:
            # Legacy unstructured file: must be a single-field [('value', ...)] entry
            assert len(dtype_str_list) == 1 and dtype_str_list[0][0] == "value", (
                f"Unstructured on-disk array but dtype attr is {dtype_str_list}"
            )
            assert dtype_str_list[0][1] == str(array.dtype), (
                f"Unstructured array dtype {array.dtype} doesn't match attr {dtype_str_list}"
            )

        # set_mask (new layout)
        if "set_mask" in ARay_grp:
            set_mask = np.asarray(ARay_grp["set_mask"])
        else:
            _warn_legacy()
            set_mask = None  # caller infers from sentinels

        # Mapping dict
        mapping_grp = ARay_grp["mapping_dict"]
        mapping_keys = mapping_grp.attrs["mapping_dict_keys"]
        if len(mapping_keys) > 0:
            mapping_dict = {}
            type_lookup = {"int": int, "float": float, "str": str}
            for key in mapping_keys:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                assert isinstance(key, str)
                mapped_types = np.asarray(mapping_grp[f"{key}-types"], dtype=str)
                mapped = np.asarray(mapping_grp[f"{key}-mapping"]["mapped"])
                idx = np.asarray(mapping_grp[f"{key}-mapping"]["index"])

                assert len(mapped_types) == len(mapped) == len(idx)

                reconstructed_mapping_for_key = {}
                for i in range(len(mapped_types)):
                    typ = type_lookup[mapped_types[i]]
                    if typ == str:
                        val = mapped[i]
                        if isinstance(val, bytes):
                            val = val.decode("utf-8")
                        reconstructed_mapping_for_key[val] = int(idx[i])
                    else:
                        reconstructed_mapping_for_key[typ(mapped[i])] = int(idx[i])

                mapping_dict[key] = reconstructed_mapping_for_key
        else:
            mapping_dict = None

    assert all(array.shape[i] == shape[i] for i in range(max(len(shape), len(array.shape))))

    return array, set_mask, mapping_dict, unset_bool, unset_int, unset_float, unset_object, vlen_names


def write_to_file(file_path, other_args):
    (array, set_mask, shape, dtype_str, mapping_dict,
     unset_bool, unset_int, unset_float, unset_object, vlen_names) = other_args

    with h5py.File(file_path, "w") as h5_file:
        ARay_grp = h5_file.require_group("ARay")

        ARay_grp.attrs["shape"] = shape
        ARay_grp.attrs["dtype"] = dtype_str
        ARay_grp.attrs["unset_bool"] = unset_bool
        ARay_grp.attrs["unset_int"] = unset_int
        ARay_grp.attrs["unset_float"] = unset_float
        ARay_grp.attrs["unset_object"] = unset_object
        ARay_grp.attrs["vlen_names"] = vlen_names

        ARay_grp.create_dataset(
            "array",
            data=array,
            dtype=array.dtype,
            compression="gzip",
            compression_opts=4,
        )
        ARay_grp.create_dataset(
            "set_mask",
            data=set_mask,
            dtype=set_mask.dtype,
            compression="gzip",
            compression_opts=4,
        )

        mapping_grp = ARay_grp.require_group("mapping_dict")
        if mapping_dict is None:
            mapping_grp.attrs["mapping_dict_keys"] = []
        else:
            mapping_grp.attrs["mapping_dict_keys"] = [str(key) for key in mapping_dict.keys()]
            vlen_str = h5py.string_dtype(encoding="utf-8")
            dt_mapping = np.dtype([("mapped", vlen_str), ("index", np.int64)])
            for key, value_dict in mapping_dict.items():
                save_types_array = np.array(
                    [type(mapped).__name__ for mapped in value_dict.keys()], dtype=vlen_str
                )
                mapping_grp.create_dataset(
                    f"{key}-types",
                    data=save_types_array,
                    dtype=vlen_str,
                    compression="gzip",
                    compression_opts=4,
                )
                save_array = np.array(
                    [(str(mapped), idx) for mapped, idx in value_dict.items()], dtype=dt_mapping
                )
                mapping_grp.create_dataset(
                    f"{key}-mapping",
                    data=save_array,
                    dtype=dt_mapping,
                    compression="gzip",
                    compression_opts=4,
                )


def list_available_files(directory, prefix: str = "") -> list:
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    for i, f in enumerate(files):
        print(f"{i} - {os.path.basename(f)}")
    return files
