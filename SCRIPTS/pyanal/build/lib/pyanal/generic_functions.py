import numpy as np
from matplotlib import pyplot as plt
import math
from numba import njit, prange
import sys

# subprocess
import os, subprocess, multiprocessing, importlib, pickle, tempfile

FASTMATH = True


def divide_defined(x, y,
                zero_over_zero=np.nan,
                inf_over_inf=np.nan,
                nan_result=np.nan,
                atol=1e-16):
    """
    Element-wise x / y, fast-pathed for the common all-finite case (numba JIT).

    Note: the sign of the output only depends on the sign of x and y. Integer values of '0' without any sign are considered positive.
    """
    xb, yb = np.broadcast_arrays(np.asarray(x, dtype=float),
                                 np.asarray(y, dtype=float))
    xf = np.ascontiguousarray(xb).ravel()
    yf = np.ascontiguousarray(yb).ravel()
    out = np.empty(xf.size, dtype=float)
    _divide_kernel(xf, yf, out, float(zero_over_zero),
                   float(inf_over_inf), float(nan_result), float(atol))
    out = out.reshape(xb.shape)
    return out.item() if out.ndim == 0 else out

def divide_x_by_zero_gets_a(x, y, a: float | np.floating):
    """
    return: x/y where y!=0 and a where y==0.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    y0 = np.isclose(y, 0)
    out = np.divide(x, y, out=np.full(np.broadcast(x, y).shape, a), where=~y0)
    return out.item() if out.ndim == 0 else out

def divide_nan(x, y):
    return divide_x_by_zero_gets_a(x, y, np.nan)

def divide_abs_nan(x, y):
    array = divide_x_by_zero_gets_a(x, y, np.nan)
    mask_nan = ~np.isnan(array)
    array[mask_nan] = np.abs(array[mask_nan])
    return array

def log_zero(x):
    return np.log(x) if x != 0 else 0

def fold_coord(coord, box_dim):
   box_dim = np.asarray(box_dim)
   coord_remain = np.remainder(coord, box_dim)
   # Numerical hygiene for strict periodic domains
   coord_remain[coord_remain < 0] = 0.0
   coord_remain[coord_remain >= box_dim - box_dim * (10 * np.finfo(float).eps) ] = 0.0
   return coord_remain

def cartesian_to_spherical_coord(coords):
    coords_array = np.asarray(coords, dtype=float)
    single = coords_array.ndim == 1
    coords_array = np.atleast_2d(coords_array)
    if coords_array.shape[1] != 3:
        raise IndexError(f"coords must be of shape (3,) or (N, 3). Got {np.asarray(coords).shape}.")
    r = np.linalg.norm(coords_array, axis=-1)
    theta = np.arccos(coords_array[:, 2] / r)
    phi = np.arctan2(coords_array[:, 1], coords_array[:, 0])
    phi[phi < 0] += 2 * np.pi
    out = np.stack((r, phi, theta), axis=-1) # radial, azimuthal, polar
    return out[0] if single else out

def min_img_dist(s, t, box_dim):
    box_dim = np.asarray(box_dim)
    s = np.asarray(s); t = np.asarray(t)
    # Ensure consistent dimensions
    if box_dim.ndim > 0 and (s.shape[-1] != t.shape[-1] or s.shape[-1] != box_dim.shape[-1]):
        raise ValueError("Last dimension of s, t, and box_dim must match")
    distance = t - s
    box_half = box_dim*0.5
    return np.remainder(distance + box_half, box_dim) - box_half

@njit(parallel=True, fastmath=FASTMATH)
def weighted_com_pbc(positions, weights, box_l):
    N, D = positions.shape
    inv_box = 1.0 / np.asarray(box_l)
    result = np.empty(D)
    for d in range(D):
        sin_sum = 0.0
        cos_sum = 0.0
        for i in prange(N):  # parallel over particles; numba reduces sin_sum/cos_sum
            angle = 6.28318530717958648 * positions[i, d] * inv_box[d] # 2*pi *
            w = weights[i]
            sin_sum += w * math.sin(angle)
            cos_sum += w * math.cos(angle)
        avg_angle = math.atan2(sin_sum, cos_sum)
        if avg_angle < 0:
            avg_angle += 6.28318530717958648 # 2pi
        result[d] = (avg_angle * 0.159154943091895336) * box_l[d] # angle * (1/2pi)

    return result

def generate_random_unit_vectors(N_PART):
    z = np.random.uniform(-1, 1, N_PART)
    r = np.sqrt(1 - z*z)
    phi = np.random.uniform(0, 2*np.pi, N_PART)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x, y, z))

##### Particle masks #####
# BULK / SURFACE analysis
def separate_bulk_surface(positions, size=1., manual_selection=False, dumb_threshold=0.):
    z = positions[:,2]

    hist, dens_z_bin_edges = np.histogram(z, bins=np.arange(0, z.max()+1+0.0001, 0.5), density=True)
    hist_avg_loco = np.zeros((len(hist),))
    for i in range(1, len(hist)-1):
        hist_avg_loco[i] = np.mean(hist[i-1:i+2])
    hist_avg_loco = np.asarray(hist_avg_loco)
    hist = np.asarray(hist)

    bin_centers = (dens_z_bin_edges[:-1] + dens_z_bin_edges[1:]) / 2

    substrate_threshold = 2*size# exclude bottom two layers
    substrate_threshold_z = substrate_threshold + size/2

    if dumb_threshold:
        bulk_threshold= dumb_threshold
    else:
        # find point that goes down 4 times in a row... é o que é
        bulk_threshold= bin_centers[len(hist_avg_loco)-4]
        for i in range(0, len(hist_avg_loco)-4):
            if bin_centers[i] < substrate_threshold_z:
                continue
            if hist_avg_loco[i] > hist_avg_loco[i+1] and hist_avg_loco[i+1] > hist_avg_loco[i+2] and hist_avg_loco[i+2] > hist_avg_loco[i+3] and hist_avg_loco[i+3] > hist_avg_loco[i+4]:
                    bulk_threshold=bin_centers[i]
                    break
    
    bulk_threshold_z = bulk_threshold - size/2

    if manual_selection:
       # Create and save the plot
        plt.figure(figsize=(10, 6), num=666)
        plt.plot(bin_centers, hist, "-", linewidth=2)
        plt.axvline(x=substrate_threshold, color='red', linestyle='--', linewidth=2, label=f"Substrate boundary ({substrate_threshold:.2f})")
        plt.axvline(x=bulk_threshold, color='orange', linestyle='--', linewidth=2,
                    label=f'Auto bulk boundary ({bulk_threshold:.2f})')
        plt.xlabel('z position')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Current Phase Separation - Check saved image')
        plt.grid(True, alpha=0.3)
        plt.savefig("foda-se.png", dpi=150, bbox_inches='tight')
        plt.close(666)

        # Text-based interface
        print(f"Available z-range: {bin_centers[0]:.2f} to {bin_centers[-1]:.2f}")

        while True:
            try:
                user_input = input("Enter manual bulk threshold (or press Enter for current threshold): ").strip()
                if not user_input:
                    print(f"Using automatic threshold: ", end="")
                    break
                else:
                    bulk_threshold = float(user_input)
                    bulk_threshold_z = bulk_threshold - size/2
                    if bulk_threshold_z < substrate_threshold_z or bulk_threshold > bin_centers[-1]:
                        print(f"Error: Threshold must be between {substrate_threshold+1:.2f} and {bin_centers[-1]:.2f}")
                        continue
                    print(f"Using manual threshold: ", end="")

                    plt.figure(figsize=(10, 6), num=666)
                    plt.plot(bin_centers, hist, "-", linewidth=2)
                    plt.axvline(x=substrate_threshold, color='red', linestyle='--', linewidth=2, label=f"Substrate boundary ({substrate_threshold:.2f})")
                    plt.axvline(x=bulk_threshold, color='orange', linestyle='--', linewidth=2,
                                label=f'Last bulk boundary ({bulk_threshold:.2f})')
                    plt.xlabel('z position')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.title('Current Phase Separation - Check saved image')
                    plt.grid(True, alpha=0.3)
                    plt.savefig("foda-se.png", dpi=150, bbox_inches='tight')
                    plt.close(666)

                    continue
            except ValueError:
                print("Error: Please enter a valid number")

    # correct for particle size
    print(f"Using diameter corrected bulk z-region: {substrate_threshold_z} <= z <= {bulk_threshold_z:.2f} (thickness {(bulk_threshold - substrate_threshold)})")

    mask_bulk = (z > substrate_threshold_z) & (z < bulk_threshold_z)
    mask_surface = z > bulk_threshold_z

    return substrate_threshold_z, bulk_threshold_z, mask_bulk, mask_surface
#####

##### Helper functions #####
@njit(cache=True, fastmath=False, parallel=False)
def _divide_kernel(x, y, out, zz, ii, nn, atol):
    inf = math.inf
    for i in prange(x.size):
        xi = x[i]; yi = y[i]
        if math.isfinite(xi) and math.isfinite(yi):
            if abs(yi) >= atol:
                out[i] = xi / yi
            else:
                s = math.copysign(1.0, xi) * math.copysign(1.0, yi)
                out[i] = (inf if abs(xi) >= atol else zz) * s
        elif math.isnan(xi) or math.isnan(yi):
            out[i] = nn
        elif math.isinf(xi) and math.isinf(yi):
            out[i] = ii * (math.copysign(1.0, xi) * math.copysign(1.0, yi))
        else:
            out[i] = xi / yi

##### System/Memmory #####
def run_subprocess_in_module(args, module, worker):
     # Create a temp file to store arguments
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        pickle.dump(args, f)
        args_path = f.name

    # Create a temp file to store the result
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        result_path = f.name

    try:
        # Build a one-liner to execute in subprocess
        cmd = (
            f"import pickle; "
            f"from {module} import {worker} as worker; "
            f"args = pickle.load(open('{args_path}', 'rb')); "
            f"res = worker(*args); "
            f"pickle.dump(res, open('{result_path}', 'wb'))"
        )

        subprocess.run([sys.executable, "-c", cmd], check=True)

        # Load the result
        with open(result_path, "rb") as f:
            result = pickle.load(f)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Worker failed for {worker} from {module} with args {args} (exit code {e.returncode})"
        )
    finally:
        # Clean up temp files
        os.remove(args_path)
        os.remove(result_path)

    return result
def tempfile_path(suffix=None, delete=False):
    tmp = tempfile.NamedTemporaryFile(delete=delete, suffix=suffix)
    tempfile_path = tmp.name
    tmp.close()
    return tempfile_path

def _mp_worker_entry(module, worker, args, conn):
    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, worker)
        res = fn(*args)
        conn.send(("ok", res))
    except BaseException as e:
        import traceback
        conn.send(("err", repr(e), traceback.format_exc()))
    finally:
        conn.close()

def run_subprocess_in_module_mp(args, module, worker):
    """Spawn-context Process variant of run_subprocess_in_module. Single-use:
    the worker process exits after one call so the OS reclaims all native
    memory (numba JIT, scipy Delaunay scratch, KDTree, pyscal/igraph C-state).
    Result is sent back over a Pipe — typically a string path to a .npz file
    the worker wrote, or a simple varibale like a bool, int or float. Exceptions in the worker are re-raised in the parent
    with the original traceback string included."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_mp_worker_entry, args=(module, worker, args, child_conn))
    p.start()
    child_conn.close()
    try:
        status = parent_conn.recv()
    except EOFError:
        p.join()
        raise RuntimeError(
            f"Worker {module}.{worker} exited without sending result (exit code {p.exitcode})"
        )
    p.join()
    parent_conn.close()
    if status[0] == "ok":
        return status[1]
    raise RuntimeError(f"Worker {module}.{worker} raised {status[1]}:\n{status[2]}")
#####

# Explicit export list
__all__ = [
    "fold_coord",
    "min_img_dist",
    "separate_bulk_surface",
    "divide_nan",
]
