import numpy as np
import time

from pyanal import ARay
from pyanal.ARay.compatibility import _fit_a_in_b

from itertools import product

# Create array structure
data_entry_dtype = [("a", "int16"), ("b", "int16"), ("c", "float32")]

# Initialize array give shape and structure
a = ARay((3,2,4), dtype=data_entry_dtype)

# Add dict like indices mappign of array
a.add_index_mapping({"ts": [100, 500, 600], "DENS": [0.2, 0.3], "H": [0, 1, 3, 6]})

data_entry_dtype = [("a", "int16"), ("b", "int16"), ("c", "float32")]

# Initialize array give shape and structure
b = ARay((4, 5, 3), dtype=data_entry_dtype)

# Add dict like indices mappign of array
b.add_index_mapping({"ts": [0, 100, 500, 600], "H": [0, 1, 3, 6, 10], "DENS": [0.2, 0.25, 0.3]})

for ts, H, DENS in product([100, 500, 600], [0, 1, 3, 6], [0.2, 0.3]):
    a.set({"a": ts, "b": H, "c": DENS}, ts=ts, DENS=DENS, H=H)

print(a)

print(a.fits_in(b), b.fits_in(a))

print(b)

_fit_a_in_b(a, b)

for ts, H, DENS in product([0, 100, 500, 600], [0, 1, 3, 6, 10], [0.2, 0.25, 0.3]):
    print(f"{ts} {H} {DENS}:", b.get(ts=ts, H=H, DENS=DENS))