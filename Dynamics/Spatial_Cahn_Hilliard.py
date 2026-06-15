"""Spatial (Model B) Cahn-Hilliard dynamics for multi-component Flory-Huggins mixtures.

All functions are for 2-D periodic-boundary PDE fields phi(x, y) with shape
(nc, nx, ny). This module is independent of Model_A.py (which handles the
well-mixed / point-composition ODE system).

Public API
----------
Core physics
  simplex_project_np        - NumPy simplex projection
  simplex_project_jax       - JAX simplex projection (jit-able)
  init_phi_multi_jax        - initialize phi with smooth noise
  fprime_multi_jax          - chemical potential field (FH + gradient)
  compute_energy_spectral_jax - total free energy via spectral method
  compute_energy_map_jax    - local energy density map
  step_jax                  - single r-stabilized semi-implicit step
  run_cahn_hilliard_multi_FH_rstab_jax_until_converged - full dynamics runner

Memory utilities
  partition_memory          - binary -> continuous memory via softmax
  hebbian_learning_matrix   - Hebbian chi from partitioned memories
  valid_vectors / generate_vectors - generate binary memory pairs
  generate_memories         - convenience wrapper
  identify_memories         - map each pixel to closest stored memory

Plotting helpers
  plot_energy               - energy vs time curve
  plot_final_components     - tiled per-component imshow
  plot_memories_grayscale   - grayscale strip view of memories
"""

import functools

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator

jax.config.update("jax_enable_x64", True)


# =======================
# Utilities / init
# =======================

def simplex_project_np(phi, eps=1e-6):
    """Clip and renormalize phi to the probability simplex (NumPy)."""
    phi = np.clip(phi, eps, 1.0 - eps)
    phi = phi / phi.sum(axis=0, keepdims=True)
    return phi


def init_phi_multi_jax(nc, nx, ny, mean_phi=None, noise_amp=0.01, seed=0, sigma_spatial=1.0):
    """Initialize phi with smooth periodic noise via a spectral Gaussian filter.

    Returns shape (nc, nx, ny) with pointwise sum_i phi_i = 1.
    """
    rng = np.random.default_rng(seed)
    if mean_phi is None:
        mean_phi = np.full(nc, 1.0 / nc, dtype=float)
    mean_phi = np.asarray(mean_phi, dtype=float).ravel()
    if mean_phi.size == 1:
        mean_phi = np.full(nc, float(mean_phi), dtype=float)
    assert mean_phi.size == nc, f"mean_phi.size={mean_phi.size} != nc={nc}"

    phi = np.zeros((nc, nx, ny), dtype=float)
    for i in range(nc):
        phi[i] = mean_phi[i] + noise_amp * (rng.random((nx, ny)) - 0.5)

    if sigma_spatial is not None and sigma_spatial > 0:
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0)
        K2 = kx[:, None] ** 2 + ky[None, :] ** 2
        filt = np.exp(-0.5 * (sigma_spatial ** 2) * K2)
        for i in range(nc):
            phi_hat = np.fft.fft2(phi[i])
            phi[i] = np.real(np.fft.ifft2(phi_hat * filt))

    phi = simplex_project_np(phi, eps=1e-6)
    return jnp.asarray(phi, dtype=jnp.float64)


# =======================
# Physics functions
# =======================

@functools.partial(jit, static_argnums=())
def simplex_project_jax(phi, eps=1e-6):
    """Clip and renormalize phi to the probability simplex (JAX, jit-able)."""
    phi = jnp.clip(phi, eps, 1.0 - eps)
    phi = phi / jnp.sum(phi, axis=0, keepdims=True)
    return phi


@functools.partial(jit, static_argnums=())
def fprime_multi_jax(phi, chi, nu=None):
    """Chemical potential field: f'_i = (1/nu_i) log(phi_i) + sum_j chi_ij phi_j.

    phi: (nc, nx, ny), chi: (nc, nc), nu: (nc,) or None -> all ones.
    """
    eps = 1e-6
    phi_safe = jnp.clip(phi, eps, 1.0 - eps)
    lp = jnp.log(phi_safe)
    entropic = lp if nu is None else lp / nu[:, None, None]
    inter = jnp.einsum("ij,jxy->ixy", chi, phi_safe)
    return entropic + inter


@functools.partial(jit, static_argnums=())
def compute_energy_spectral_jax(phi, chi, kappa, K2, dx=1.0, dy=1.0, nu=None):
    """Total Flory-Huggins free energy with gradient penalty (spectral method).

    F = integral [ sum_i phi_i/nu_i log(phi_i)
                 + 1/2 sum_ij chi_ij phi_i phi_j
                 + 1/2 sum_i kappa_i |grad phi_i|^2 ] dr
    """
    nc, nx, ny = phi.shape
    area_element = dx * dy
    N = nx * ny

    eps = 1e-6
    phi_safe = jnp.clip(phi, eps, 1.0)

    if nu is None:
        mix_term = jnp.sum(phi_safe * jnp.log(phi_safe)) * area_element
    else:
        mix_term = jnp.sum((phi_safe / nu[:, None, None]) * jnp.log(phi_safe)) * area_element

    I = jnp.einsum("ixy,jxy->ij", phi_safe, phi_safe) * area_element
    inter_term = 0.5 * jnp.sum(chi * I)

    phi_hat = jnp.fft.fft2(phi_safe, axes=(1, 2))
    power = phi_hat.real ** 2 + phi_hat.imag ** 2
    grad_term = 0.5 * jnp.sum(
        kappa[:, None, None] * (area_element / N) * (K2[None, :, :] * power)
    )

    return mix_term + inter_term + grad_term


def compute_energy_map_jax(phi, chi, kappa, K2, dx=1.0, dy=1.0, nu=None):
    """Local free energy density map.

    Returns
    -------
    energy_map  : (nx, ny)  mixing + interaction density
    grad_map    : (nx, ny)  gradient energy density
    total_map   : (nx, ny)  sum of both
    """
    nc, nx, ny = phi.shape
    area_element = dx * dy
    N = nx * ny

    eps = 1e-6
    phi_safe = jnp.clip(phi, eps, 1.0)

    if nu is None:
        mix_map = jnp.sum(phi_safe * jnp.log(phi_safe), axis=0)
    else:
        mix_map = jnp.sum((phi_safe / nu[:, None, None]) * jnp.log(phi_safe), axis=0)

    inter_map = 0.5 * jnp.einsum("ij,ixy,jxy->xy", chi, phi_safe, phi_safe)
    energy_map = mix_map + inter_map

    phi_hat = jnp.fft.fft2(phi_safe, axes=(1, 2))
    lap_phi = jnp.fft.ifft2(-K2[None, :, :] * phi_hat, axes=(1, 2)).real
    grad_map = -0.5 * jnp.sum(kappa[:, None, None] * phi_safe * lap_phi, axis=0)

    return energy_map, grad_map, energy_map + grad_map


# =======================
# Jitted one-step update
# =======================

@functools.partial(jit, static_argnums=())
def step_jax(phi, dt, M, kappa, chi, nu, K2, K4, r_stab):
    """One r-stabilized semi-implicit step in Fourier space.

    Returns
    -------
    phi_raw  : raw inverse-FFT field before clipping/renormalization
    phi_proj : projected field after positivity clipping and renormalization
    """
    fprime = fprime_multi_jax(phi, chi, nu)

    phi_hat = jnp.fft.fft2(phi, axes=(1, 2))
    fhat = jnp.fft.fft2(fprime, axes=(1, 2))

    mu0_hat = fhat + kappa[:, None, None] * (K2[None, :, :] * phi_hat)

    sumM = jnp.sum(M)
    weighted_mu0 = jnp.tensordot(M, mu0_hat, axes=(0, 0))
    lambda_hat = -weighted_mu0 / sumM
    lambda_hat = lambda_hat.at[0, 0].set(0.0 + 0.0j)

    f_explicit = fhat - r_stab * phi_hat

    numer = phi_hat - dt * (
        M[:, None, None] * K2[None, :, :] * (f_explicit + lambda_hat[None, :, :])
    )
    denom = 1.0 + dt * (
        M[:, None, None] * (r_stab * K2[None, :, :] + kappa[:, None, None] * K4[None, :, :])
    )
    denom = denom.astype(phi_hat.dtype)

    phi_hat_new = numer / denom
    phi_raw = jnp.real(jnp.fft.ifft2(phi_hat_new, axes=(1, 2)))
    phi_proj = simplex_project_jax(phi_raw)

    return phi_raw, phi_proj


# =======================
# Runner: until converged
# =======================

def run_cahn_hilliard_multi_FH_rstab_jax_until_converged(
    nc=3, nx=128, ny=128, dx=1.0, dy=1.0,
    M=None, kappa=None, chi=None, nu=None,
    dt=0.05, mean_phi=None, initial_phi=None,
    noise_amp=0.02, save_interval=200, seed=None,
    verbose=True, track_energy=True, training=False,
    r_stab=None, max_steps=500000, max_time=500.0,
    dt_min=1e-6, dt_max=None,
    energy_increase_tol=1e-6, mean_tol=1e-5,
    raw_mass_tol=1e-5, raw_min_tol=-1e-6,
    field_tol=1e-6, energy_slope_tol=1e-7,
    window=20, min_steps_check=1000,
    grow_dt=1.05, sigma_spatial=1.0,
):
    """Run Cahn-Hilliard dynamics until convergence (r-stabilized semi-implicit).

    When `training=True` skips frame saving and returns only the final phi array.
    Otherwise returns (frames, phi, mean_phi_global, energies, steps_saved, time_saved).
    """
    if M is None:
        M = jnp.ones(nc, dtype=jnp.float64)
    else:
        M = jnp.asarray(M, dtype=jnp.float64)
        assert M.size == nc

    if kappa is None:
        kappa = jnp.ones(nc, dtype=jnp.float64)
    else:
        kappa = jnp.asarray(kappa, dtype=jnp.float64)
        assert kappa.size == nc

    if chi is None:
        chi = jnp.zeros((nc, nc), dtype=jnp.float64)
    else:
        chi = jnp.asarray(chi, dtype=jnp.float64)
        assert chi.shape == (nc, nc)

    if nu is not None:
        nu = jnp.asarray(nu, dtype=jnp.float64)
        assert nu.size == nc

    if r_stab is None:
        r_stab = float(jnp.max(jnp.abs(chi))) + 1.0

    if initial_phi is not None:
        phi = simplex_project_jax(jnp.asarray(initial_phi, dtype=jnp.float64))
    else:
        phi = init_phi_multi_jax(
            nc=nc, nx=nx, ny=ny, mean_phi=mean_phi,
            noise_amp=noise_amp,
            seed=0 if seed is None else seed,
            sigma_spatial=sigma_spatial,
        )

    mean_phi_global = np.array(phi.mean(axis=(1, 2)))

    if dt_max is None:
        dt_max = dt

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    K4 = K2 ** 2

    frames, energies, steps_saved, time_saved = [], [], [], []
    t = 0.0
    step = 0
    n_accept = 0
    n_reject = 0
    max_slope = 0.0

    if track_energy:
        E_old = float(compute_energy_spectral_jax(phi, chi, kappa, K2, dx, dy, nu))
        recent_energies = [E_old]
        recent_times = [t]
    else:
        E_old = None
        recent_energies = []
        recent_times = []

    if not training:
        frames.append(np.array(phi))
        steps_saved.append(step)
        if track_energy:
            energies.append(E_old)
            time_saved.append(t)

    while True:
        if step >= max_steps:
            print("Maximum accepted-step limit reached.")
            break
        if t >= max_time:
            print("Maximum time limit reached.")
            break
        if dt < dt_min:
            print("dt fell below dt_min.")
            break

        dt_try = min(dt, dt_max)
        phi_raw, phi_new = step_jax(phi, dt_try, M, kappa, chi, nu, K2, K4, r_stab)

        min_raw = float(jnp.min(phi_raw))
        raw_mass_err = float(jnp.max(jnp.abs(jnp.sum(phi_raw, axis=0) - 1.0)))
        mean_phi_new = np.array(phi_new.mean(axis=(1, 2)))

        if track_energy:
            E_new = float(compute_energy_spectral_jax(phi_new, chi, kappa, K2, dx, dy, nu))
        else:
            E_new = None

        reject = (
            min_raw < raw_min_tol
            or raw_mass_err > raw_mass_tol
            or np.max(np.abs(mean_phi_new - mean_phi_global)) > mean_tol
            or (track_energy and E_new - E_old > energy_increase_tol)
        )

        if reject:
            dt *= 0.5
            n_reject += 1
            continue

        rel_change = float(
            jnp.max(jnp.abs(phi_new - phi)) / (jnp.max(jnp.abs(phi)) + 1e-12) / dt_try
        )
        phi = phi_new
        t += dt_try
        step += 1
        n_accept += 1

        if track_energy:
            E_old = E_new
            recent_energies.append(E_new)
            recent_times.append(t)
            if len(recent_energies) > window:
                recent_energies.pop(0)
                recent_times.pop(0)

        dt = min(grow_dt * dt_try, dt_max)

        if training:
            continue

        if step % save_interval == 0:
            frames.append(np.array(phi))
            steps_saved.append(step)
            if track_energy:
                energies.append(E_old)
                time_saved.append(t)
            if verbose:
                var_by_comp = [float(np.var(np.array(phi[i]))) for i in range(nc)]
                print(
                    f"time={t:.6f}, step={step}, dt={dt_try:.3e}, "
                    f"rel_change={rel_change:.3e}, min_raw={min_raw:.3e}, "
                    f"raw_mass_err={raw_mass_err:.3e}, "
                    f"mean={np.array(phi.mean(axis=(1, 2)))}, var={var_by_comp}"
                    + (f", E={E_old:.6e}" if track_energy else "")
                )

        if step >= min_steps_check:
            energy_slope_avg = None
            if track_energy and len(recent_energies) >= 2:
                dE = recent_energies[-2] - recent_energies[-1]
                dt_window = recent_times[-1] - recent_times[-2]
                energy_slope_avg = (
                    jnp.abs(dE / (recent_energies[-1] * dt_window))
                    if dt_window > 0 else np.inf
                )
                max_slope = max(max_slope, energy_slope_avg)

            converged_field = rel_change < field_tol
            converged_energy = (
                energy_slope_avg is not None
                and energy_slope_tol is not None
                and energy_slope_avg < energy_slope_tol * max_slope
            )

            if converged_energy:
                print(f"Energy slope converged. Final rel_change: {rel_change:.3e}, "
                      f"max_slope: {max_slope:.3e}, slope: {energy_slope_avg:.3e}")
                break
            if converged_field:
                print(f"Field converged. Final rel_change: {rel_change:.3e}")
                if energy_slope_avg is not None:
                    print(f"Energy slope: {energy_slope_avg:.3e}")
                break

    if training:
        return np.array(phi)

    print(f"Accepted steps: {n_accept}, Rejected steps: {n_reject}, Final time: {t:.6f}")
    return (
        np.array(frames),
        np.array(phi),
        mean_phi_global,
        np.array(energies) if track_energy else None,
        np.array(steps_saved),
        np.array(time_saved) if track_energy else None,
    )


# =======================
# Memory utilities
# =======================

def partition_memory(memory, beta):
    """Convert binary memory to continuous composition via softmax with temperature beta."""
    memory = np.exp(np.array(memory) / beta)
    return memory / np.sum(memory)


def hebbian_learning_matrix(target_memories, memory_ratio):
    """Build Hebbian interaction matrix chi from partitioned target memories."""
    N = len(target_memories[0])
    chi = np.zeros((N, N))
    for i, memory in enumerate(target_memories):
        chi -= np.outer(memory, memory) * memory_ratio[i]
    np.fill_diagonal(chi, 0)
    return chi


def valid_vectors(N, k1, k2, overlap):
    """Check whether (N, k1, k2, overlap) parameters are feasible."""
    return 0 <= overlap <= min(k1, k2) and k1 + k2 - overlap <= N


def generate_vectors(N, k1, k2, overlap, seed=None):
    """Generate a pair of binary vectors of length N with given bit counts and overlap."""
    if seed is not None:
        np.random.seed(seed)
    if not (0 <= overlap <= min(k1, k2)):
        raise ValueError("Invalid overlap: must be between 0 and min(k1, k2)")
    if k1 + k2 - overlap > N:
        raise ValueError("Too many 1s requested for given overlap and N")

    indices = np.arange(N)
    np.random.shuffle(indices)
    shared_ones = indices[:overlap]
    remaining = indices[overlap:]
    a_extra = remaining[:(k1 - overlap)]
    b_extra = remaining[(k1 - overlap):(k1 - overlap) + (k2 - overlap)]

    A = np.zeros(N, dtype=int)
    B = np.zeros(N, dtype=int)
    A[shared_ones] = 1
    A[a_extra] = 1
    B[shared_ones] = 1
    B[b_extra] = 1
    return [A, B]


def generate_memories(N, i, j, beta=0.5, plot=True, seed=42):
    """Generate a pair of binary memories, optionally extend and partition them."""
    memories = generate_vectors(N, i, i, j, seed=seed)
    target_size = N + 1
    extended = []
    for mem in memories:
        if mem.shape[0] < target_size:
            mem = np.append(mem, np.zeros(target_size - mem.shape[0]))
        extended.append(mem)

    p_memories = [partition_memory(m, beta) for m in extended]
    if plot:
        plot_memories_grayscale(p_memories)
    return p_memories


def identify_memories(Fin_phi, memories):
    """Map each pixel to its closest stored memory.

    Parameters
    ----------
    Fin_phi   : (nx, ny, M) array of per-pixel composition vectors
    memories  : (K, M) array of reference memories

    Returns
    -------
    average_vectors  : (K, M) mean composition per memory region
    figure           : matplotlib Figure
    closest_indices  : (nx, ny) int array of closest memory index
    min_distances    : (nx, ny) float array of L2 distance to closest memory
    """
    N = Fin_phi.shape[0]
    K, M = np.shape(memories)[0], np.shape(memories)[1]

    distances = np.linalg.norm(
        Fin_phi[:, :, np.newaxis, :M] - memories[:M], axis=3
    )
    closest_indices = np.argmin(distances, axis=2)
    min_distances = np.min(distances, axis=2)

    base_colors = plt.cm.get_cmap("tab20", K)
    cmap_custom = ListedColormap(base_colors(np.arange(K)))

    figure = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(closest_indices, cmap=cmap_custom, vmin=0, vmax=K - 1)
    plt.xticks([])
    plt.yticks([])
    cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cbar1.set_label("Closest Memory Index")
    cbar1.set_ticks(np.arange(K))
    cbar1.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Closest Memory Mapping")
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(min_distances, cmap="viridis")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im2, label="Min Distance", fraction=0.046, pad=0.04)
    plt.title("Minimum Distance Heatmap")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    average_vectors = np.zeros((K, M))
    counts = np.zeros(K)
    for i in range(N):
        for j in range(N):
            idx = closest_indices[i, j]
            average_vectors[idx] += Fin_phi[i, j]
            counts[idx] += 1
    nonzero = counts > 0
    average_vectors[nonzero] /= counts[nonzero, np.newaxis]
    plot_memories_grayscale(average_vectors)

    return average_vectors, figure, closest_indices, min_distances


# =======================
# Plotting helpers
# =======================

def plot_energy(times, energies, title="Total Free Energy vs Time"):
    """Plot total free energy over time."""
    plt.figure(figsize=(6, 4))
    plt.plot(times, energies, marker="o", linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Total Free Energy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_components(phi_final, titles=None, figsize_per_panel=3.5):
    """Tiled per-component imshow of a final phi field."""
    nc = phi_final.shape[0]
    if titles is None:
        titles = [f"phi_{i}" for i in range(nc)]
    fig, axes = plt.subplots(
        1, nc, figsize=(figsize_per_panel * nc, figsize_per_panel), squeeze=False
    )
    axes = axes[0]
    for i in range(nc):
        im = axes[i].imshow(np.array(phi_final[i]), origin="lower")
        axes[i].set_title(titles[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_memories_grayscale(memories, titles=None, filename=None):
    """Plot memories as grayscale strips with a shared colorbar."""
    n_memories = len(memories)
    n_components = len(memories[0])
    fig, axes = plt.subplots(
        n_memories, 1, figsize=(n_components, n_memories),
        gridspec_kw={"hspace": 0.4},
    )
    if n_memories == 1:
        axes = [axes]

    all_values = np.concatenate(memories)
    vmin, vmax = np.min(all_values), np.max(all_values)

    for i, memory in enumerate(memories):
        print(f"Memory {i + 1}: {memory}")
        ax = axes[i]
        im = ax.imshow(memory.reshape(1, -1), cmap="gray_r", aspect="auto", vmin=vmin, vmax=vmax)
        if titles and len(titles) > i:
            ax.set_title(titles[i], loc="left", pad=10)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel(f"Memory {i + 1}", rotation=0, ha="right", labelpad=20)
        if i < n_memories - 1:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks(np.arange(n_components))
            ax.set_xticklabels([str(j) for j in range(n_components)])
            ax.set_xlabel("Component Index")

    fig.colorbar(im, ax=axes, label="Volume Fraction", fraction=0.2, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
