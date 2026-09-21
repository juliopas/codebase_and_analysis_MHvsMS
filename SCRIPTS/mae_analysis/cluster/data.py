"""Cluster ARay creation / selection (mirrors surface/data.py)."""

import json
import os

from pyanal import ARay

from .config import (
    do_FLAGS, do_CALCULATE, l_list,
    ts_list, DENS_list, K_list, HEIGHT_list, H_list,
    DATA_ARAY_DIR,
)


def init_ARay(A, use_file_id=None, user_input=True) -> tuple[ARay, str]:
    """Create (or load) the labelled cluster ARay for one model `A`."""
    data_entry_dtype = ARay.create_data_entry_dtype(l_list=l_list, **do_FLAGS)

    data_ARay = ARay((len(ts_list), len(DENS_list), len(K_list), len(HEIGHT_list), len(H_list)),
                     dtype=data_entry_dtype)
    data_ARay.add_index_mapping({"ts": ts_list, "DENS": DENS_list, "K": K_list,
                                 "HEIGHT": HEIGHT_list, "H": H_list})

    filename_data_ARay = f"{A}--cluster"

    if do_CALCULATE:
        data_ARay, DATA_ARAY_FILENAME = data_ARay.match_aray_from_file(
            filename=filename_data_ARay + ".h5", directory=DATA_ARAY_DIR,
            try_file_id=use_file_id, user_input=user_input)
    else:
        data_ARay, DATA_ARAY_FILENAME = choose_data_ARay_from_file(
            directory=DATA_ARAY_DIR, _filename_0=filename_data_ARay + ".h5", use_file_id=use_file_id)
    assert isinstance(DATA_ARAY_FILENAME, str)
    return data_ARay, DATA_ARAY_FILENAME


_LAST_PICK_JSON = ".mae_aray_last_pick.json"


def choose_data_ARay_from_file(directory="", _filename_0: str ="", use_file_id=None) -> tuple[ARay, str]:
    files = [f for f in os.listdir(directory) if f.startswith(_filename_0)]
    if len(files) < 1:
        raise RuntimeError(f"No ARay file found in {directory} starting with {_filename_0}")
    len_files = len(files)

    if use_file_id is not None:
        assert isinstance(use_file_id, int) and 0 <= use_file_id < len_files
        file_name = files[use_file_id]
        return ARay.from_file(os.path.join(directory, file_name)), file_name

    if len_files == 1:
        print(f"Auto-selected: {files[0]}")
        return ARay.from_file(os.path.join(directory, files[0])), files[0]

    # Determine default: last pick for this prefix if still present, else newest by mtime
    default = None
    try:
        json_path = os.path.join(directory, _LAST_PICK_JSON)
        with open(json_path, "r") as f:
            last = json.load(f).get(_filename_0)
        if last in files:
            default = files.index(last)
    except Exception:
        pass
    if default is None:
        default = max(range(len_files), key=lambda i: os.path.getmtime(os.path.join(directory, files[i])))

    for i, f in enumerate(files):
        marker = " [default]" if i == default else ""
        print(f"{i} - {os.path.basename(f)}{marker}")

    while True:
        raw = input(f"pick 0...{len_files-1} [default {default}]: ").strip()
        if raw == "":
            choice = default
            break
        try:
            choice = int(raw)
            if 0 <= choice < len_files:
                break
        except ValueError:
            pass

    file_name = files[choice]
    try:
        json_path = os.path.join(directory, _LAST_PICK_JSON)
        data = {}
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception:
            pass
        data[_filename_0] = file_name
        with open(json_path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return ARay.from_file(os.path.join(directory, file_name)), file_name
