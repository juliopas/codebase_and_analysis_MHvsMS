# pyanal.ARay.compatibility

import warnings

import numpy as np

##### Test array compatibility #####
def _a_fits_in_b(a, b):
    # mapping compatibility
    if set(a.mapping_dict.keys()) != set(b.mapping_dict.keys()):
        return False
    if not all([set(a.mapping_dict[key]).issubset(b.mapping_dict[key]) for key in a.mapping_dict]):
        return False

    # dtype compatibility
    a_dtype_str = a.dtype_str
    b_dtype_str = b.dtype_str
    if isinstance(a_dtype_str, str):
        if isinstance(b_dtype_str, str):
            return a_dtype_str == b_dtype_str
        else:
            return a_dtype_str in b_dtype_str
    else:
        return all(name_type_tuple in b_dtype_str for name_type_tuple in a_dtype_str)
#####

##### Explain why two arays are incompatible #####
def _trunc(items, n=8):
    """Render a list compactly, truncating long ones."""
    items = list(items)
    if len(items) <= n:
        return ", ".join(map(str, items))
    return ", ".join(map(str, items[:n])) + f", ... (+{len(items) - n} more)"


def _describe_incompatibility(data_ARay, file_aray):
    """Return a short, human-readable report of why `data_ARay` and `file_aray`
    do not fit into each other.

    Reports, in both directions: axis keys, axis labels (per shared key), and
    dtype fields present in one aray but missing from the other.
    """
    lines = []

    # ---- axis keys ----
    keys_new = set(data_ARay.mapping_dict.keys()) if data_ARay.mapping_dict else set()
    keys_file = set(file_aray.mapping_dict.keys()) if file_aray.mapping_dict else set()
    only_new = keys_new - keys_file
    only_file = keys_file - keys_new
    if only_new:
        lines.append(f"  - axes only in new aray: {_trunc(sorted(only_new))}")
    if only_file:
        lines.append(f"  - axes only in file:     {_trunc(sorted(only_file))}")

    # ---- axis labels (per shared key) ----
    for key in sorted(keys_new & keys_file):
        labels_new = set(data_ARay.mapping_dict[key])
        labels_file = set(file_aray.mapping_dict[key])
        miss_in_file = labels_new - labels_file
        miss_in_new = labels_file - labels_new
        if miss_in_file:
            lines.append(f"  - axis '{key}' labels only in new aray: {_trunc(sorted(miss_in_file))}")
        if miss_in_new:
            lines.append(f"  - axis '{key}' labels only in file:     {_trunc(sorted(miss_in_new))}")

    # ---- dtype fields ----
    def _field_names(dtype_str):
        if isinstance(dtype_str, str):
            return {dtype_str}
        return {name for name, _ in dtype_str}

    fields_new = _field_names(data_ARay.dtype_str)
    fields_file = _field_names(file_aray.dtype_str)
    only_new_f = fields_new - fields_file
    only_file_f = fields_file - fields_new
    if only_new_f:
        lines.append(f"  - fields only in new aray: {_trunc(sorted(only_new_f))}")
    if only_file_f:
        lines.append(f"  - fields only in file:     {_trunc(sorted(only_file_f))}")

    if not lines:
        lines.append("  - (axes/labels/fields match; incompatible dtypes or types differ)")

    return "Incompatibility:\n" + "\n".join(lines)
#####

##### Add compatible aray data #####
def _fit_a_in_b(a, b):
    """Copy `a`'s set entries into `b` at the cells with matching axis labels.

    This *overwrites* `b` at the destination cells (both value and set_mask):
    callers normally pass a freshly-constructed, all-unset `b`, so nothing is
    clobbered. If any destination cell is already set in `b` and also set in `a`,
    a warning is emitted reporting how many were overwritten.
    """
    assert _a_fits_in_b(a, b)

    ndim_a = a.ndim
    ndim_b = b.ndim
    assert ndim_a == ndim_b

    names = [name_type_tuple[0] for name_type_tuple in a._dtype_str]
    axes_a = a.mapping_dict.keys()
    b_axis_idx = {key: i for i, key in enumerate(b.mapping_dict.keys())}
    # ---------- build translators (vectorized over labels) ----------
    translators = {}
    for key in axes_a:
        map_a = a.mapping_dict[key]
        map_b = b.mapping_dict[key]

        idx_a = np.array(list(map_a.values()))
        # destination indices in same order
        idx_b = np.array([map_b[l] for l in map_a])

        translator = np.empty(len(idx_a), dtype=np.intp)
        translator[idx_a] = idx_b

        translators[key] = translator
    # ---------- build destination advanced index arrays ----------
    dst_indices = [None] * ndim_b
    for ax, key in enumerate(axes_a):
        shape = [1] * ndim_a
        shape[ax] = a.shape[ax]

        src_axis = np.arange(a.shape[ax]).reshape(shape)
        dst_axis = translators[key][src_axis]

        dst_indices[b_axis_idx[key]] = dst_axis

    dst_indices = tuple(dst_indices)

    # Warn if the merge overwrites cells already set in the destination.
    overlap = 0
    for name in names:
        overlap += int(np.count_nonzero(b._set_mask[name][dst_indices] & a._set_mask[name]))
    if overlap:
        warnings.warn(f"_fit_a_in_b overwrote {overlap} already-set destination entries", stacklevel=2)

    # Assign per-field to avoid relying on view semantics of multi-field indexing,
    # which has changed across numpy versions. Also propagate set_mask.
    for name in names:
        b._array[name][dst_indices] = a._array[name]
        b._set_mask[name][dst_indices] = a._set_mask[name]

#####

def _match_aray_to_file_aray(data_ARay, file_aray, filename, directory, _filename_0, _count, files, try_file_id=None, user_input=True):
    if _a_fits_in_b(data_ARay, file_aray):
        # return the file aray
        return file_aray, filename
    elif _a_fits_in_b(file_aray, data_ARay):
        if try_file_id is not None:
            _fit_a_in_b(file_aray, data_ARay)
            return data_ARay, filename
        elif user_input:
            action, new_name, = _ask_user_to_fit(filename, _filename_0)
            if action == "fit":
                _fit_a_in_b(file_aray, data_ARay)
                return data_ARay, new_name

    if try_file_id is not None and not user_input:
        raise FileExistsError()

    if user_input:
        # incompatible → ask once
        action, new_name, new_count = _ask_user_overwrite(_filename_0, _count, files,
                                                          data_ARay, file_aray)
    else:
        action = "new"
        new_count = _count + 1
        new_name  = f"{_filename_0}-{new_count}.h5"

    if action == "overwrite":
        return data_ARay, filename

    # new filename
    return data_ARay.match_aray_from_file(filename=new_name, directory=directory,
                               user_input=user_input, _filename_0=_filename_0,
                               _count=new_count)


def _ask_user_overwrite(prefix, count, files, data_ARay=None, file_aray=None):
    new_count = count + 1

    print("Incompatible ARay file: the new aray does not fit the file and the file "
          "does not fit the new aray.")
    if data_ARay is not None and file_aray is not None:
        print(_describe_incompatibility(data_ARay, file_aray))
    print("  overwrite -> write a FRESH aray over the file; the old file's values "
          "are NOT copied and will be LOST.")
    print(f"  save-as   -> keep the old file, write the new aray to "
          f"'{prefix}-{new_count}.h5'.")

    ans = input("Choose (overwrite/save-as/custom/pick): ").strip()

    if ans == "overwrite":
        return "overwrite", None, None

    if ans == "save-as":
        return "new", f"{prefix}-{new_count}.h5", new_count
    # non-destructive alternatives
    if ans == "custom":
        s = input(f"custom ({prefix}-[x].h5): ")
        return "new", f"{prefix}-{s}.h5", count
    if ans == "pick":
        len_files = len(files)
        if len_files == 1:
            print(f"Auto-selected: {files[0]}")
            return "new", files[0], count
        for i, f in enumerate(files):
            print(f"{i} - {f}")
        while True:
            raw = input(f"pick 0...{len_files-1}: ").strip()
            try:
                choice = int(raw)
                if 0 <= choice < len_files:
                    break
            except ValueError:
                pass
        return "new", files[choice], count

    raise InterruptedError()

def _ask_user_to_fit(filename, prefix):
    ans = input(
        f"aray does not fit into file aray, but file aray fits into aray. Do you want to copy all contents of file aray into the new aray? (yes/no): "
    )
    if ans == "yes":
        ans = input(
            f"Overwrite the file with name {filename}? (yes/no): "
        )
        if ans == "yes":
            return "fit", filename
        else:
            ans = input(
                f"Input name to save to ({prefix}-[input string]): "
            )
            return "fit", f"{prefix}-{ans}.h5"

    elif ans == "no":
        return "dont_fit", None

    raise InterruptedError()
