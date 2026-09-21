from numba import njit, prange

import numpy as np
import math

FASTMATH = True # use False for slightly more precise results. Significantly slower on larger systems for dipolar energy calculations with direct sum

##### Main functions #####
def calc_energies(box_l, ids, poss, dips, bonds, H_ext, periodicity=[True, True, True], p3m=False):
    box_l = np.asarray(box_l) + np.asarray(~np.asarray(periodicity), dtype=int) * 1E6
    H_ext = np.asarray(H_ext)

    n_part = poss.shape[0]

    assert n_part == dips.shape[0] == len(bonds)
    # assert (np.linalg.norm(dips, axis=-1) > 0).all() or np.linalg.norm(H_ext) == 0.

    if p3m:
        raise NotImplementedError("Best course of actions is probably make take the espresso c++ functions and make a little c++ class for p3m , tha tI can import to python.")
        E_dip = dipolar_energy_p3m(box_l, poss, dips)
    else:
        E_dip = dipolar_energy_direct_sum(box_l, poss, dips, n_part)
    E_Zee = Zeeman_energy(dips, n_part, H=H_ext)
    E_Zee_dipm = Zeeman_dipm_energy(dips, n_part, H=H_ext)

    # Bond energy requires to re-format bonds array
    bond_ends, bond_idxs, bond_ks, bond_r0s = convert_bonds_list_for_numba(ids, bonds, n_part)
    E_bond = bond_energy_harmonic(box_l, n_part, poss, bond_ends, bond_idxs, bond_ks, bond_r0s)

    return E_dip / n_part, E_Zee / n_part, E_bond / n_part, E_Zee_dipm / n_part
#####

##### Dipolar Energy #####
## Direct sum ## (Not properly implemented. See espresso. Maybe make c++ addon to library)
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def dipolar_energy_direct_sum(box_l, positions, dipoles, N):
    inv_box0 = 1.0 / box_l[0]
    inv_box1 = 1.0 / box_l[1]
    inv_box2 = 1.0 / box_l[2]

    energy = 0.0
    for i in prange(N):
        ri0, ri1, ri2 = positions[i]
        mui0, mui1, mui2 = dipoles[i]

        e_i = 0.0
        for j in range(i+1, N):
            rj0, rj1, rj2 = positions[j]

            r0 = rj0 - ri0
            r1 = rj1 - ri1
            r2 = rj2 - ri2

            # ---- minimum image convention (FAST) ----
            r0 -= box_l[0] * math.floor(r0 * inv_box0 + 0.5)
            r1 -= box_l[1] * math.floor(r1 * inv_box1 + 0.5)
            r2 -= box_l[2] * math.floor(r2 * inv_box2 + 0.5)

            r2sq = r0*r0 + r1*r1 + r2*r2
            if r2sq < 1e-6:
                continue

            inv_r = 1.0 / math.sqrt(r2sq)
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r2 * inv_r
            inv_r5 = inv_r3 * inv_r2

            muj0, muj1, muj2 = dipoles[j]

            mu_dot = mui0*muj0 + mui1*muj1 + mui2*muj2
            mur_i = mui0*r0 + mui1*r1 + mui2*r2
            mur_j = muj0*r0 + muj1*r1 + muj2*r2

            e_i += mu_dot * inv_r3 - 3.0 * mur_i * mur_j * inv_r5

        energy += e_i

    return energy
##

## P3M ##
def dipolar_energy_p3m(box_l, positions, dipoles, z_scale=5, r_cut_prefactor=0.45, alpha_prefactor=5., mesh_prefactor=2):
    box_l_local = box_l.copy()
    box_l_local[2] = z_scale*min(box_l)

    r_cut = r_cut_prefactor * min(box_l_local)
    alpha = alpha_prefactor / r_cut
    mesh = tuple(int(mesh_prefactor*L/r_cut) for L in box_l_local)

    N = dipoles.shape[0]

    E_real = dipolar_real_p3m(box_l_local, positions, dipoles, N, alpha, r_cut)
    E_rec  = dipolar_recip_p3m(box_l_local, positions, dipoles, N, alpha, mesh)
    E_self = dipolar_self_energy(dipoles, N, alpha)
    E_Yeh = slab_correction_dipoles(dipoles, N, box_l_local)

    return E_real + E_rec + E_self + E_Yeh
# E real space
@njit(parallel=True, fastmath=FASTMATH)
def dipolar_real_p3m(box_l, positions, dipoles, N, alpha, r_cut, prefactor=1.):
    energy = 0.0

    invL0 = 1.0 / box_l[0]
    invL1 = 1.0 / box_l[1]
    invL2 = 1.0 / box_l[2]

    rcut2 = r_cut * r_cut
    alpha_sqr = alpha * alpha
    coeff = 2.0 * alpha * 0.56418958354775629  # 2*alpha/sqrt(pi)

    for i in prange(N):
        ri0, ri1, ri2 = positions[i]
        mui0, mui1, mui2 = dipoles[i]

        ei = 0.0
        for j in range(i+1, N):
            rj0, rj1, rj2 = positions[j]

            dx = rj0 - ri0
            dy = rj1 - ri1
            dz = rj2 - ri2

            dx -= box_l[0] * math.floor(dx*invL0 + 0.5)
            dy -= box_l[1] * math.floor(dy*invL1 + 0.5)
            dz -= box_l[2] * math.floor(dz*invL2 + 0.5)

            r2 = dx*dx + dy*dy + dz*dz
            if r2 > rcut2 or r2 < 1e-6:
                continue

            r = math.sqrt(r2)
            invr = 1.0 / r
            invr2 = invr * invr

            erfc = math.erfc(alpha * r)

            muj0, muj1, muj2 = dipoles[j]

            mu_dot = mui0*muj0 + mui1*muj1 + mui2*muj2
            mur_i = mui0*dx + mui1*dy + mui2*dz
            mur_j = muj0*dx + muj1*dy + muj2*dz

            expf = math.exp(-alpha_sqr*r2)

            A = (erfc * invr + coeff * expf) * invr2
            B = (3.0 * A + 2.0 * alpha_sqr * coeff * expf) * invr2

            ei += prefactor * (mu_dot * A - mur_i*mur_j * B)

        energy += ei

    return energy

# E reciprocal space
def dipolar_recip_p3m(box_l, positions, dipoles, N, alpha, mesh):
    Lx, Ly, Lz = box_l
    Mx, My, Mz = mesh
    V = Lx * Ly * Lz

    # dipole density on grid
    rho = np.zeros((Mx, My, Mz, 3), dtype=np.float64)

    # ---- assignment (NGP for simplicity; order-4 is similar but longer) ----
    assign_dipoles_bspline4_numba(rho, positions, dipoles, N, box_l, mesh)

    # FFT
    rho_k = np.fft.fftn(rho, axes=(0,1,2))

    # k-vectors
    kx = 2*np.pi*np.fft.fftfreq(Mx, d=Lx/Mx)
    ky = 2*np.pi*np.fft.fftfreq(My, d=Ly/My)
    kz = 2*np.pi*np.fft.fftfreq(Mz, d=Lz/Mz)

    # vectorize
    kxv = kx[:, None, None]   # shape (Mx,1,1)
    kyv = ky[None, :, None]   # shape (1,My,1)
    kzv = kz[None, None, :]   # shape (1,1,Mz)

    k2 = kxv**2 + kyv**2 + kzv**2
    mask = k2 != 0.0           # skip zero vector

    # k·mu
    kdotmu = (
        rho_k[:,:,:,0]*kxv +
        rho_k[:,:,:,1]*kyv +
        rho_k[:,:,:,2]*kzv
    )

    invW2 = assignment_factor(kx, ky, kz, box_l, mesh, p=4)
    inv4a2 = 1.0 / (4*alpha*alpha)
    Gk = np.zeros_like(k2)
    Gk[mask] = np.exp(-k2[mask] * inv4a2) / k2[mask]

    energy = np.sum(Gk * (kdotmu.real*kdotmu.real + kdotmu.imag*kdotmu.imag) * invW2) / (2.0 * V)

    return energy
# helpers for E reciprocal space
@njit(inline='always', fastmath=FASTMATH)
def bspline4(u):
    u_sqr = u * u
    u_cub = u_sqr * u
    inv_6 = 0.16666666666666667
    u_cub_2 = u_cub * 0.5
    u_cub_6 = u_cub * inv_6
    u_sqr_2 = u_sqr * 0.5
    u_2 = u * 0.5

    w0 = inv_6 + u_sqr_2 - u_2 - u_cub_6
    w1 = u_cub_2 - u_sqr + 0.66666666666666667
    w2 = u_sqr_2 + u_2 - u_cub_2 + inv_6
    w3 = u_cub_6
    return w0, w1, w2, w3
@njit(fastmath=FASTMATH)
def assign_dipoles_bspline4_numba(rho, positions, dipoles, N, box_l, mesh):
    Lx, Ly, Lz = box_l
    Mx, My, Mz = mesh

    invLx = Mx / Lx
    invLy = My / Ly
    invLz = Mz / Lz

    for pi in range(N):
        p0, p1, p2 = positions[pi]
        gx = p0 * invLx
        gy = p1 * invLy
        gz = p2 * invLz

        ix = int(math.floor(gx)) - 1
        iy = int(math.floor(gy)) - 1
        iz = int(math.floor(gz)) - 1

        ux = gx - (ix + 1)
        uy = gy - (iy + 1)
        uz = gz - (iz + 1)

        wx0, wx1, wx2, wx3 = bspline4(ux)
        wy0, wy1, wy2, wy3 = bspline4(uy)
        wz0, wz1, wz2, wz3 = bspline4(uz)

        mu0, mu1, mu2 = dipoles[pi]

        wx = (wx0, wx1, wx2, wx3)
        wy = (wy0, wy1, wy2, wy3)
        wz = (wz0, wz1, wz2, wz3)

        for dx in range(4):
            i = (ix + dx) % Mx
            wxi = wx[dx]
            for dy in range(4):
                j = (iy + dy) % My
                wxy = wxi * wy[dy]
                for dz in range(4):
                    k = (iz + dz) % Mz
                    w = wxy * wz[dz]
                    rho[i,j,k,0] += mu0 * w
                    rho[i,j,k,1] += mu1 * w
                    rho[i,j,k,2] += mu2 * w
def sinc(x):
    out = np.ones_like(x)
    mask = x != 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out
def assignment_factor(kx, ky, kz, box_l, mesh, p=4):
    dx = box_l[0] / mesh[0]
    dy = box_l[1] / mesh[1]
    dz = box_l[2] / mesh[2]

    Wx = sinc(0.5 * kx * dx)**p
    Wy = sinc(0.5 * ky * dy)**p
    Wz = sinc(0.5 * kz * dz)**p

    invW2 = 1.0 / (Wx[:,None,None] * Wy[None,:,None] * Wz[None,None,:])**2

    return invW2

# E self corection
@njit(parallel=True, fastmath=FASTMATH)
def dipolar_self_energy(dipoles, N, alpha):
    energy = 0.0
    for i in prange(N):
        mu0, mu1, mu2 = dipoles[i]
        energy += (mu0*mu0 + mu1*mu1 + mu2*mu2)
    energy *= -(2.0 * alpha*alpha*alpha) * 0.18806319451591876
    return energy

# E dipolar layer correction
@njit(parallel=True, fastmath=FASTMATH)
def slab_correction_dipoles(dipoles, N, box_l, factor=1):
    box0, box1, box2 = box_l
    V_inv = 1 / (box0 * box1 * box2)
    Mz = 0.0
    for i in prange(N):
        mu2 = dipoles[i,2]
        Mz += mu2
    energy = 6.2831853071795864 * Mz*Mz * V_inv * factor # const is 2*pi
    return energy
##

## Dipolar energy, assuming one long chain ##
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def dipolar_energy_chains(dipoles, particle_size):
    # average energy if all chained
    N = dipoles.shape[0]

    mu_sqr = 0.0
    for i in prange(N):
        mui0, mui1, mui2 = dipoles[i]
        mui_sqr = mui0*mui0 + mui1*mui1 + mui2*mui2

        mu_sqr += mui_sqr
    inv_sigma = 1 / particle_size
    energy = - 2 * (mu_sqr / N) * inv_sigma*inv_sigma*inv_sigma

    return energy
##

##### Bonded energy #####
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def bond_energy_harmonic(box_l, N, positions, bond_ends, bond_ids, bond_ks, bond_r0s):
    inv_box0 = 1.0 / box_l[0]
    inv_box1 = 1.0 / box_l[1]
    inv_box2 = 1.0 / box_l[2]

    # particle 0: bond_start = 0
    energy = 0.0
    ri0, ri1, ri2 = positions[0]
    for j in range(0, bond_ends[0]):
        id_j = bond_ids[j]
        rj0, rj1, rj2 = positions[id_j]
        rij0 = rj0 - ri0; rij1 = rj1 - ri1; rij2 = rj2 - ri2
        rij0 -= box_l[0] * math.floor(rij0 * inv_box0 + 0.5)
        rij1 -= box_l[1] * math.floor(rij1 * inv_box1 + 0.5)
        rij2 -= box_l[2] * math.floor(rij2 * inv_box2 + 0.5)
        r_ij = math.sqrt(rij0*rij0 + rij1*rij1 + rij2*rij2)
        delta_r = r_ij - bond_r0s[j]
        energy += 0.5 * bond_ks[j] * (delta_r*delta_r)

    for i in prange(1, N):
        ri0, ri1, ri2 = positions[i]
        bond_start_i = bond_ends[i-1]
        bond_end_i = bond_ends[i]
        energy_i = 0.0
        for j in range(bond_start_i, bond_end_i):
            id_j = bond_ids[j]
            k_j = bond_ks[j]
            r0_ij = bond_r0s[j]

            rj0, rj1, rj2 = positions[id_j]

            rij0 = rj0 - ri0
            rij1 = rj1 - ri1
            rij2 = rj2 - ri2

            # ---- minimum image convention (FAST) ----
            rij0 -= box_l[0] * math.floor(rij0 * inv_box0 + 0.5)
            rij1 -= box_l[1] * math.floor(rij1 * inv_box1 + 0.5)
            rij2 -= box_l[2] * math.floor(rij2 * inv_box2 + 0.5)

            r_ij = math.sqrt(rij0*rij0 + rij1*rij1 + rij2*rij2)

            delta_r = (r_ij - r0_ij)

            energy_i += 0.5 * k_j * (delta_r*delta_r)

        energy += energy_i

    return energy
#####

##### Zeeman energy #####
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def Zeeman_energy(dipoles, N, H):
    H0, H1, H2 = H

    energy = 0.0
    for i in prange(N):
        mui0, mui1, mui2 = dipoles[i]

        energy -= mui0*H0 + mui1*H1 + mui2*H2

    return energy

## Zeeman energy special
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def Zeeman_dipm_energy(dipoles, N, H):
    H0, H1, H2 = H

    energy = 0.0
    for i in prange(N):
        mui0, mui1, mui2 = dipoles[i]
        dipm = mui0*mui0 + mui1*mui1 + mui2*mui2

        if dipm < 1e-6:
                continue

        inv_dipm = 1.0 / math.sqrt(dipm)

        energy -= (mui0*H0 + mui1*H1 + mui2*H2) * inv_dipm

    return energy
##
#####

##### Input helper funtions #####
## format input bonds to numba friendly numpy arrays
def convert_bonds_list_for_numba(ids, bonds, n_part):
    ids_local = ids
    bonds_local = bonds

    id_to_idx = {ids_local[i]: i for i in range(n_part)}

    total_bonds = sum(len(bonds_local[i]) for i in range(n_part))

    bond_ends = np.empty(n_part, dtype=np.int32)
    bond_idxs = np.empty(total_bonds, dtype=np.int32)
    bond_ks   = np.empty(total_bonds, dtype=np.float32)
    bond_r0s  = np.empty(total_bonds, dtype=np.float32)

    idx = 0
    for i in range(n_part):
        bonds_i = bonds_local[i]
        for bond in bonds_i:
            bond_idxs[idx] = id_to_idx[bond[4]]
            bond_ks[idx]   = bond[1]
            bond_r0s[idx]  = bond[2]
            idx += 1
        bond_ends[i] = idx

    return bond_ends, bond_idxs, bond_ks, bond_r0s
##
#####
